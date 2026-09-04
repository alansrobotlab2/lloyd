#!/usr/bin/env python3
"""Concurrent batch runner for the v4 relation classifier.

Drives `classify_edge_v4` from `classify-relationships-v4.py` over many
edges in parallel via a thread pool. Resumable: edges already present in
any `classified-v4*.jsonl` (including this run's output) are skipped, so
a kill + restart picks up where it left off.

Defaults:
- Filter: active `mentions` edges only
- Concurrency: 4 (safe ceiling — higher has co-OOM-killed with vLLM)
- Output:    _pipeline/memory-graph/classified-v4-batch.jsonl

Usage:
  .venvs/lloyd/bin/python scripts/memory/classify-v4-batch.py
  .venvs/lloyd/bin/python scripts/memory/classify-v4-batch.py --sample 20
  .venvs/lloyd/bin/python scripts/memory/classify-v4-batch.py --concurrency 6 --all-types
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
V4_PATH = SCRIPTS_DIR / "classify-relationships-v4.py"

# Hyphen in filename → load by path
_spec = importlib.util.spec_from_file_location("classify_relationships_v4", str(V4_PATH))
_v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v4)
classify_edge_v4 = _v4.classify_edge_v4
_v2 = _v4._v2  # _load_relationships, _load_fact_snippets, defaults

CLASSIFIED_DIR = Path.home() / "lloyd" / "_pipeline" / "memory-graph"
DEFAULT_OUTPUT = CLASSIFIED_DIR / "classified-v4-batch.jsonl"

# Provenance values whose `mentions` edges are eligible for re-typing.
# Anything else (STATED, INFERRED, EXTRACTED_CLASSIFIER*, …) is preserved
# as-is — those carry human or prior-classifier intent we shouldn't override.
ELIGIBLE_PROVENANCES = frozenset({"EXTRACTED"})

_stop = threading.Event()


def _context_hash(context: str) -> str:
    """Stable fingerprint of the fact-snippet context the LLM saw.

    SHA1 is fine — collisions don't matter for cache invalidation, and
    short hex (40 chars) keeps the JSONL records readable. Empty context
    hashes deterministically too so 'no facts available' is its own
    cacheable state.
    """
    return hashlib.sha1((context or "").encode("utf-8")).hexdigest()


def _install_signal_handlers() -> None:
    """First SIGINT/SIGTERM → drain (cancel pending, finish in-flight).
    Second → hard exit."""
    def _handler(signum, _frame):
        if not _stop.is_set():
            print(f"\n[signal] {signum} received — draining; second signal aborts hard",
                  file=sys.stderr)
            _stop.set()
        else:
            print("\n[signal] second signal — aborting", file=sys.stderr)
            sys.exit(130)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _load_existing_records() -> dict[tuple[str, str], str | None]:
    """Map every (source, target) pair to its recorded context_hash from
    any `classified-v4*.jsonl`. Pairs without a `context_hash` field
    (legacy records from before fingerprinting) map to None and resume
    treats them as up-to-date.

    On duplicate pairs (same pair across multiple files or runs), the
    last-seen `context_hash` wins."""
    by_pair: dict[tuple[str, str], str | None] = {}
    for fpath in sorted(CLASSIFIED_DIR.glob("classified-v4*.jsonl")):
        try:
            with fpath.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    s, t = r.get("source"), r.get("target")
                    if not s or not t:
                        continue
                    by_pair[(s, t)] = r.get("context_hash")
        except OSError as e:
            print(f"[warn] could not read {fpath}: {e}", file=sys.stderr)
    return by_pair


def _build_context_and_hash(
    edge: dict, max_ctx_chars: int,
) -> tuple[str, str]:
    """Load fact snippets for an edge and hash them. Returns (context, hash).
    Runs in worker threads so the disk reads happen in parallel."""
    context = _v2._load_fact_snippets(
        edge["source"], edge["target"], max_ctx_chars,
    )
    return context, _context_hash(context)


def _classify_one(
    edge: dict,
    context: str,
    context_hash: str,
    *,
    endpoint: str,
    model: str,
    timeout: int,
    skip_direction_check: bool,
) -> tuple[dict, dict, str, float]:
    """Run the v4 classifier on a single edge with pre-built context.

    Returns (edge, result, context_hash, elapsed_ms)."""
    t0 = time.perf_counter()
    result = classify_edge_v4(
        edge["source"], edge["target"], context,
        endpoint, model, timeout,
        skip_direction_check=skip_direction_check,
    )
    return edge, result, context_hash, (time.perf_counter() - t0) * 1000


def _prepare_candidate(
    edge: dict,
    prior: dict[tuple[str, str], str | None],
    max_ctx_chars: int,
) -> tuple[dict, str, str] | None:
    """Decide whether `edge` needs classification this run.

    Returns (edge, context, context_hash) if we should classify, or None
    if the prior record is up-to-date and we can skip. This is what the
    worker pool calls — disk I/O happens in parallel, and the main thread
    just filters None vs (edge, ctx, hash).
    """
    context, ch = _build_context_and_hash(edge, max_ctx_chars)
    key = (edge["source"], edge["target"])
    if key in prior:
        prior_hash = prior[key]
        # Legacy records without a recorded hash map to None — treat as
        # up-to-date so old data doesn't trigger a stampede on first run
        # of the new logic. New classifications always carry a hash; over
        # time the legacy entries are replaced naturally if facts change
        # in ways that flag them.
        if prior_hash is None or prior_hash == ch:
            return None
    return edge, context, ch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Cap number of edges processed (after dedup)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--endpoint", default=_v2.DEFAULT_ENDPOINT)
    p.add_argument("--model", default=_v2.DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=_v2.DEFAULT_TIMEOUT_SEC)
    p.add_argument("--max-ctx-chars", type=int, default=_v2.DEFAULT_MAX_CTX_CHARS)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Concurrent LLM calls (safe ceiling ~4)")
    p.add_argument("--all-types", action="store_true",
                   help="Classify all edge types, not just `mentions`")
    p.add_argument("--only-types", default=None,
                   help="Comma-separated edge types to classify")
    p.add_argument("--no-direction-check", action="store_true",
                   help="Skip the second LLM call that verifies direction")
    p.add_argument("--no-resume", action="store_true",
                   help="Don't filter out edges already in classified-v4*.jsonl")
    args = p.parse_args()

    if args.all_types and args.only_types:
        print("[error] --all-types and --only-types are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.concurrency < 1:
        print("[error] --concurrency must be >= 1", file=sys.stderr)
        return 2

    _install_signal_handlers()

    print(f"[info] endpoint={args.endpoint} model={args.model} "
          f"concurrency={args.concurrency}")

    all_edges = _v4._kg_store().edges.all()

    # Filter by type. Provenance gate (mentions only) applies in default
    # mode but not for explicit --all-types/--only-types power-user runs.
    if args.only_types:
        only_set = {t.strip() for t in args.only_types.split(",") if t.strip()}
        candidates = [e for e in all_edges
                      if not e.get("expired_at") and e.get("type") in only_set]
    elif args.all_types:
        candidates = [e for e in all_edges if not e.get("expired_at")]
    else:
        candidates = [
            e for e in all_edges
            if not e.get("expired_at")
            and e.get("type") == "mentions"
            and (e.get("provenance") or "") in ELIGIBLE_PROVENANCES
        ]
    print(f"[info] {len(candidates)} candidate edges (post type+provenance filter)")

    # Resume map: (source, target) -> prior context_hash (or None for legacy
    # records without a hash). Pairs we've classified before with matching
    # context are skipped; pairs whose context fingerprint changed get
    # re-classified so reclassification follows fact churn instead of
    # being frozen at first sight.
    prior: dict[tuple[str, str], str | None] = {}
    if not args.no_resume:
        prior = _load_existing_records()
        print(f"[info] {len(prior)} pairs in prior classified-v4*.jsonl")

    if args.sample:
        candidates = candidates[: args.sample]
        print(f"[info] capped to {len(candidates)} (--sample)")

    if not candidates:
        print("[info] nothing to do.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_lock = threading.Lock()
    out_fh = args.output.open("a")

    ok = 0
    fail = 0
    skipped = 0
    vocab_counts: Counter = Counter()
    adjust_counts: Counter = Counter()
    total = len(candidates)
    t_start = time.perf_counter()

    classify_kwargs = dict(
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        skip_direction_check=args.no_direction_check,
    )

    def _process_one(edge: dict) -> dict:
        """Worker: build context + hash, skip if prior matches, else classify.

        Returns one of:
          {action: 'skip',       edge}
          {action: 'classified', edge, result, context_hash, dt_ms}
          {action: 'fail',       edge, dt_ms, reason}
        """
        prep = _prepare_candidate(edge, prior, args.max_ctx_chars)
        if prep is None:
            return {"action": "skip", "edge": edge}
        edge, context, ch = prep
        t0 = time.perf_counter()
        try:
            result = classify_edge_v4(
                edge["source"], edge["target"], context,
                args.endpoint, args.model, args.timeout,
                skip_direction_check=args.no_direction_check,
            )
        except Exception as exc:  # noqa: BLE001
            return {"action": "fail", "edge": edge,
                    "dt_ms": (time.perf_counter() - t0) * 1000,
                    "reason": repr(exc)}
        return {"action": "classified", "edge": edge, "result": result,
                "context_hash": ch, "dt_ms": (time.perf_counter() - t0) * 1000}

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency,
                                thread_name_prefix="classify") as pool:
            futures = {pool.submit(_process_one, e): e for e in candidates}
            completed = 0
            for fut in as_completed(futures):
                if _stop.is_set():
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                completed += 1
                try:
                    out = fut.result()
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    edge = futures[fut]
                    print(f"  [{completed}/{total}] EXC "
                          f"{edge['source']!r}->{edge['target']!r}: {exc!r}",
                          file=sys.stderr)
                    continue

                action = out["action"]
                edge = out["edge"]

                if action == "skip":
                    skipped += 1
                    if skipped <= 3 or skipped % 500 == 0:
                        print(f"  [{completed}/{total}] SKIP (cached)         "
                              f"     {edge['source'][:22]!r:<26}->"
                              f"{edge['target'][:22]!r:<26}")
                    continue

                if action == "fail":
                    fail += 1
                    print(f"  [{completed}/{total}] FAIL              "
                          f"{out['dt_ms']:5.0f}ms "
                          f"{edge['source'][:22]!r:<26}->"
                          f"{edge['target'][:22]!r:<26}: {out.get('reason','')}",
                          file=sys.stderr)
                    continue

                # action == "classified"
                result = out["result"]
                ctx_hash = out["context_hash"]
                dt_ms = out["dt_ms"]
                new_type = result.get("type")
                if new_type not in _v4.VOCABULARY:
                    fail += 1
                    print(f"  [{completed}/{total}] FAIL              "
                          f"{dt_ms:5.0f}ms "
                          f"{edge['source'][:22]!r:<26}->"
                          f"{edge['target'][:22]!r:<26}",
                          file=sys.stderr)
                    continue

                ok += 1
                vocab_counts[new_type] += 1
                adj = result.get("verdict_adjustment", "none")
                adjust_counts[adj] += 1

                record = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "resolved_src": result.get("resolved_src", edge["source"]),
                    "resolved_tgt": result.get("resolved_tgt", edge["target"]),
                    "original_type": edge.get("type", ""),
                    "original_provenance": edge.get("provenance", ""),
                    "new_type": new_type,
                    "confidence": result.get("confidence"),
                    "reason": result.get("reason"),
                    "src_type_hint": result.get("src_type_hint"),
                    "tgt_type_hint": result.get("tgt_type_hint"),
                    "reason_quote": result.get("reason_quote"),
                    "quote_verified": result.get("quote_verified"),
                    "direction_check": result.get("direction_check"),
                    "verdict_adjustment": adj,
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "prompt_version": "v4",
                    "context_hash": ctx_hash,
                }
                line = json.dumps(record) + "\n"
                with out_lock:
                    out_fh.write(line)
                    out_fh.flush()

                flag = "" if adj == "none" else f"[{adj}]"
                print(f"  [{completed}/{total}] {new_type:<12} "
                      f"c={result.get('confidence') or 0:.2f} {dt_ms:5.0f}ms "
                      f"{edge['source'][:22]!r:<26}->"
                      f"{edge['target'][:22]!r:<26} {flag}")
    finally:
        out_fh.close()

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 72)
    print(f"Classified: {ok} ok, {fail} failed, {skipped} skipped (cache hit) in {elapsed:.1f}s")
    if ok > 0:
        rate = ok / elapsed
        print(f"Rate: {rate:.2f} edges/s ({rate * 60:.0f}/min) "
              f"over {args.concurrency} workers")
    print("\nVocab distribution:")
    for t, c in vocab_counts.most_common():
        pct = c / max(1, ok) * 100
        print(f"  {c:>5}  ({pct:5.1f}%)  {t}")
    print("\nVerdict adjustments:")
    for a, c in adjust_counts.most_common():
        print(f"  {c:>5}  {a}")
    print(f"\nWrote -> {args.output}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
