#!/usr/bin/env python3
"""Concurrency sweep against a live vLLM engine: measure the curve, pin the config.

WHY THIS FILE EXISTS IN THIS SHAPE

Lloyd's concurrency envelope is hand-set constants: the worker pool's `slots: 6`
(`config.yaml:1630`, whose own comment justifies the number by incidents — "two
autocode rounds and two autotriage turns", then "6 since 2026-09-18: autocode went
to 4 rounds"), the KV gate's `max_kv_usage: 0.60` (`config.yaml:1642`), and the
engine's `MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"`
(`agent-services/bin/start-qwen38-flash-next.sh:356`, passed as `--max-num-seqs` at
`:640`). The only structural derivation anywhere in the tree is
`tests/test_loop_depth.py`'s `slots >= rounds + triages + 1`, which is a
queue-starvation floor and says nothing about throughput. Nobody has measured the
engine's saturation curve, so no one can say whether the operating point sits left
of the knee, on it, or past it.

That measurement was previously impossible to run honestly, for two reasons this
file fixes rather than papers over:

  1. It asked a RETIRED model for its work. The module used to carry
     `MODEL = "Qwen3.6-35B-A3B-nvfp4"`, a model `:8096` no longer serves; the
     override flag made it easy to miss. The id now comes from
     `GET <base>/v1/models` and a `--model` that the served list does not contain is
     an error, not a silent request that would have 404'd on a different box.
  2. It swept two fixed batch levels and reported nothing about the conditions of
     the measurement. A curve is per-model, per-precision and per-config, so a
     reading without the config is not a result — the workshop this method comes
     from lost its own final comparison to a config swap it only noticed afterwards.
     Every run therefore writes an artifact that pins the serving process's OWN
     argv, read out of `/proc`, including `--max-num-seqs`, `--speculative-config`
     and `--kv-cache-dtype`, and refuses to write if that argv cannot be read.

WHAT GETS REPORTED PER LEVEL

Aggregate generation tokens/s (completion tokens over the window from the first
token of any request to the last), TTFT mean and p95, and mean inter-token
latency — each repeated `--repeats` times (default 3) and reported as a mean PLUS
the max-minus-min spread across those repeats. The spread is not decoration: the
same stage stated a ±2 tok/s run-to-run noise floor, so a single-run delta under a
few percent is not a result, and a knee called from one pass is a coin flip.

Contamination is decided, not assumed. A background thread samples
`vllm:num_requests_running + num_requests_waiting` every 0.5 s for the whole sweep
(`bench-flash-next.py`'s monitor, same reason: on 2026-09-08 a "single-stream"
measurement of this slot read 50-100 tok/s against a recorded 182 because the
worker pool had started jobs underneath it, and nothing in the numbers said so). A
level whose observed in-flight peak exceeded the level it launched is
`contaminated` and is EXCLUDED from the knee table — it goes in
`excluded_levels` with its peak, so a reader sees the gap rather than a curve with
a hole in it.

The engine's own counters are snapshotted before and after every level
(`kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`,
`num_preemptions_total`), because the preemption delta is the difference between
"the engine batched it" and "the engine thrashed and recompute-paid for it".

Prefix-cache REUSE is measured, not asserted: the per-level delta of
`vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` is written into
every level row as `prefix_cache_during` beside those four. A level whose engine
serves neither counter gets `null` for both, which is a different fact from "reused
nothing" and is not recorded as one. A ratio is published only on a non-zero queries
delta, and only under the condition `bench-prefix-reuse.py:11-25` records — hits over
queries is a token-level reuse rate ONLY while
`prefix_cache_queries_total == prompt_tokens_total` — so the raw pair is always
readable even where a percentage would have been a guess.

POOLS ARE NOT ASKED NICELY

`--pool drained` (default) requires `/api/workers/status` to report
`pool.in_flight_count == 0`; `--pool paused` requires `pool.paused` to be TRUE as
the database sees it, not merely asserted by the caller — a pause whose setter died
holds the fleet across a reboot, which is the 2026-10-09 incident, so the claim is
checked and not trusted. Either way the check runs BEFORE the first request (do not
load a production engine that is not quiet) and AGAIN at write time. A failed check
writes nothing and exits 2.

PROMPT SHAPE, AND WHAT IT DOES NOT MEASURE

TWO SHAPES, AND WHAT EACH ONE MEASURES

Default shape (`shape: short` on every row): eight distinct prompts, cycled per
request slot, recorded in the artifact's `prompt_policy`. This is a short-request
ENGINE-CAPACITY probe (~`--max-tokens`, default 128, against prompts of ~130 tokens);
Lloyd's real turns are long-prefill, and nothing here characterises them. Prefix
caching is left at whatever the engine was booted with — the pinned argv is where to
see that — and because the eight prompts differ, cross-request reuse within a level is
limited to the repeats of a level, which is the policy the table is quoted under.
Reuse is nonetheless MEASURED per level, so the claim is the engine's, not this
file's.

`--long-prompt-file PATH` (`shape: long_prefill` on every row): the ladder again, but
one real captured Lloyd turn prompt — the file's contents, verbatim, on every request —
so the curve is the prefill-shaped one `bench-flash-next.py`'s single-shot
`phase_prefill` cannot give (it has no concurrency ladder). That mode is deliberately
narrow:

  * each level runs EXACTLY ONCE regardless of `--repeats`, and the warm-up pass is
    skipped: reuse engages after one cold pass (bench-prefix-reuse.py trap 2), so a
    warm-up would turn the only measured pass into a warm prefill;
  * `prompt_policy` names the file and the prompt's token count AS THE ENGINE COUNTS
    IT, via the same `/tokenize` route `bench-admission-stall.py:107` posts to (an
    estimate would be an invention, and a refusal to write beats one);
  * an n=1 row cannot state a max-minus-min spread, so it carries none — no
    `{key}_spread` key at all, rather than a `0.0` that reads as measured-zero
    variance — and it prints in its own table headed with `n=1`, never in the short
    table the knee is read from.

Usage:
  bench-35b-concurrency.py [--levels 1,2,4,6,8,12,16] [--repeats 3]
                           [--base-url http://127.0.0.1:8096] [--model <id>]
                           [--pool drained|paused] [--label pre]
                           [--out DIR] [--proc-root /proc]
                           [--long-prompt-file PATH]

The knee verdict is not computed here. This file produces the table and the
conditions; the sentence about where the envelope sits is written from a drained
window by whoever ran it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASE = "http://127.0.0.1:8096"
DEFAULT_POOL_STATUS = "http://127.0.0.1:8080/api/workers/status"

#: Engine flags a concurrency curve is a function of. Anything here is a
#: confound if two sweeps disagree about it, so all of it goes in the header —
#: `max_num_seqs` included, which is the one flag the source measurement moved
#: to turn 800 tok/s into 1,100 at concurrency 64.
ENGINE_FLAGS = (
    "max-num-seqs", "max-num-batched-tokens", "gpu-memory-utilization",
    "kv-cache-dtype", "kv-cache-memory-bytes", "swap-space",
    "enable-prefix-caching", "no-enable-prefix-caching", "async-scheduling",
    "scheduling-policy", "max-model-len", "tensor-parallel-size",
    "pipeline-parallel-size", "data-parallel-size", "speculative-config",
    "quantization",
)

#: Counters read before and after every level, and sampled between by the monitor.
ENGINE_COUNTERS = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
)

#: The two counters that say how much of a level's prefill the engine already had.
#: NOT part of `ENGINE_COUNTERS`: that tuple is the four the contamination and
#: preemption logic reads, and the monitor samples it every 0.5 s. These two are
#: snapshotted once around each level and published as a delta (see `prefix_delta`).
#: Their semantics are carried from `agent-services/bin/bench-prefix-reuse.py:11-25`.
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PROMPT_TOKENS = "vllm:prompt_tokens_total"

#: The condition under which hits/queries IS a token-level reuse rate. Recorded by the
#: sibling tool: on an MTP boot the queries counter summed across KV-cache groups, so
#: one 60k-token prompt logged 673,179 queries (11.2x) and the ratio read ~68% while
#: real cross-request reuse was zero. A ratio published without this line attached is
#: a number that can be true by accident, so it is only ever printed with it.
PREFIX_RATE_CONDITION = (
    f"token-level reuse rate ONLY while {PREFIX_QUERIES} == {PROMPT_TOKENS} "
    "(1:1 parity; bench-prefix-reuse.py:11-25) — otherwise hits/queries is "
    "cross-group arithmetic, not reuse")

#: The two prompt shapes this client can sweep. `short` is the original behaviour and
#: the table the knee is read from; `long_prefill` is n=1 per level and printed apart.
SHAPE_SHORT = "short"
SHAPE_LONG = "long_prefill"

#: The two counters that decide whether speculative decoding is running AT ALL.
#: Kept public because `eval/lloyd_profile.md` standing problem 7 states this
#: derivation in prose and `tests/test_lloyd_profile_doc_claims.py` recomputes it.
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_DRAFTED = "vllm:spec_decode_num_draft_tokens_total"

#: Distinct prompts so prefix caching does not collapse a level into one cached
#: stream. All ~130 tokens and sized to elicit a few hundred tokens.
PROMPTS = [
    "Explain how transformer attention works in exactly four sentences. " * 8,
    "Describe the tradeoffs between MoE and dense language models in four sentences. " * 8,
    "What's the difference between greedy and beam-search decoding? Four sentences. " * 8,
    "Explain prompt caching in vLLM in exactly four sentences. " * 8,
    "Describe how KV-cache memory scales with context length. Four sentences. " * 8,
    "Explain speculative decoding and MTP in four sentences. " * 8,
    "How does prefill differ from decode in an LLM server? Four sentences. " * 8,
    "Describe how FP4 quantization differs from INT4. Four sentences. " * 8,
]

PROMPT_POLICY = (
    f"{len(PROMPTS)} distinct prompts cycled across the slots of a level; a level "
    f"above {len(PROMPTS)} reuses them within the level. Short-request engine "
    "capacity probe, not Lloyd's long-prefill workload."
)


class SweepError(RuntimeError):
    """A refusal the operator has to see: wrong model, busy pool, unreadable flags."""


def http_get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def served_model_ids(base_url: str, timeout: float = 10.0) -> list[str]:
    """The ids `GET <base>/v1/models` offers, in the order the engine lists them."""
    body = json.loads(http_get(f"{base_url.rstrip('/')}/v1/models", timeout))
    return [m["id"] for m in body.get("data", []) if m.get("id")]


def resolve_model(base_url: str, requested: str | None = None,
                  timeout: float = 10.0) -> dict:
    """Pick the model to drive from the ENGINE's list, never from this file.

    A `--model` the server does not serve is a refusal: the retired default this
    replaced did not fail loudly, it just described a machine that no longer exists.
    """
    ids = served_model_ids(base_url, timeout)
    if not ids:
        raise SweepError(f"{base_url}/v1/models served no model ids")
    if requested:
        if requested not in ids:
            raise SweepError(
                f"--model {requested!r} is not what {base_url}/v1/models serves "
                f"({', '.join(ids)}); the sweep would measure an error page")
        chosen = requested
    else:
        chosen = ids[0]
    return {"model": chosen, "served_ids": ids}


def metrics_text(base_url: str, timeout: float = 5.0) -> str:
    return http_get(f"{base_url.rstrip('/')}/metrics", timeout).decode("utf-8", "replace")


def counter_sums(text: str, names: tuple[str, ...] = ENGINE_COUNTERS) -> dict[str, float]:
    """Sum every label series of each named counter — a gauge is per KV cache group.

    Script-side on purpose: `app.vllm_metrics.parse_prometheus` is the dashboard's
    stateful reader and this process is not the dashboard. The names matched here
    are the four this file is written about, not a translation table.
    """
    want = set(names) | {SPEC_ACCEPTED, SPEC_DRAFTED}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in want:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return out


#: What is snapshotted around each level: the four the contamination logic reads,
#: plus the prefix-cache pair whose delta goes in the level row beside them.
LEVEL_COUNTERS = ENGINE_COUNTERS + (PREFIX_QUERIES, PREFIX_HITS)


def engine_snapshot(base_url: str) -> dict:
    """The engine's own reading, before or after a level.

    A counter the engine does not serve is `None`, never `0.0`: `counter_sums` only
    returns names it saw, and a level recorded as "0 queries" on an engine that
    publishes neither counter would be read as zero reuse rather than no evidence.
    """
    c = counter_sums(metrics_text(base_url), LEVEL_COUNTERS)
    return {
        "kv_cache_usage": c.get("vllm:kv_cache_usage_perc"),
        "num_requests_running": c.get("vllm:num_requests_running"),
        "num_requests_waiting": c.get("vllm:num_requests_waiting"),
        "preemptions_total": c.get("vllm:num_preemptions_total"),
        "prefix_cache_queries_total": c.get(PREFIX_QUERIES),
        "prefix_cache_hits_total": c.get(PREFIX_HITS),
    }


def prefix_delta(before: dict, after: dict) -> dict:
    """The engine's prefix-cache counters moved across one level, as a delta.

    Published as the raw pair first. The ratio is a separate, conditional claim:
    `bench-prefix-reuse.py:11-25` records that hits/queries is a token-level reuse
    rate ONLY while the queries counter equals `prompt_tokens_total`, and on an MTP
    boot that parity broke by 11.2x while the percentage still read ~68%. So the
    ratio appears only for a level with a non-zero queries delta and a hits reading,
    and it travels with the condition attached; anywhere else the pair alone is what
    the artifact publishes.
    """
    def moved(key: str) -> float | None:
        b, a = before.get(key), after.get(key)
        return (a - b) if (b is not None and a is not None) else None

    queries = moved("prefix_cache_queries_total")
    hits = moved("prefix_cache_hits_total")
    out: dict = {"queries_delta": queries, "hits_delta": hits}
    if queries and hits is not None:
        out["hit_rate"] = hits / queries
        out["hit_rate_condition"] = PREFIX_RATE_CONDITION
    return out


def spec_decode_counts(base_url: str) -> dict:
    """Accepted and drafted speculative tokens, as the engine counts them.

    Public because `eval/lloyd_profile.md` standing problem 7 claims speculative
    decoding's status in prose, and the guard recomputes this pair rather than
    trusting a number written into a document: lifetime counters, so the ratio is
    readable but a quoted figure rots on the next boot.
    """
    c = counter_sums(metrics_text(base_url), (SPEC_ACCEPTED, SPEC_DRAFTED))
    accepted, drafted = c.get(SPEC_ACCEPTED), c.get(SPEC_DRAFTED)
    return {
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "accepted_over_drafted": (accepted / drafted) if drafted else None,
    }


def http_post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize_count(base_url: str, model: str, prompt: str,
                   timeout: float = 60.0) -> int:
    """How many tokens the ENGINE says a prompt is, not how many this file guesses.

    The engine's `/tokenize` route — the same one `bench-admission-stall.py:107`
    posts to (vLLM serves `/tokenize`; there is no `/v1/tokenize` on this box). A
    long-prefill curve is a statement about prefill size, so the size has to come
    from the thing that will run the prefill: there is no tokenizer helper anywhere
    else in the tree, and chars/4 would be an invention written into an artifact as a
    measurement. An engine that will not tokenize is a refusal, not a blank.
    """
    try:
        body = http_post_json(f"{base_url.rstrip('/')}/tokenize",
                              {"model": model, "prompt": prompt}, timeout)
    except Exception as exc:                                    # noqa: BLE001
        raise SweepError(
            f"{base_url}/tokenize did not answer ({exc}); the long-prefill artifact "
            "would name a prompt whose size nobody measured, so the sweep refuses to "
            "write one") from exc
    try:
        return int(body["count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SweepError(
            f"{base_url}/tokenize answered {body!r}, which carries no 'count'") from exc


def read_long_prompt(path_str: str) -> str:
    """The captured turn, verbatim. An unreadable or empty file is a refusal.

    Verbatim is the point: the shape being measured is one specific prompt, so the
    bytes on the wire are the bytes in the file — not stripped, not truncated, not
    the eight built-in prompts if this one happens to be short.
    """
    path = Path(path_str)
    if not path.is_file():
        raise SweepError(f"--long-prompt-file {path_str} is not a file; a long-shape "
                         "run whose prompt cannot be read would label a row with a "
                         "prompt that never reached the engine")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SweepError(f"--long-prompt-file {path_str} cannot be read as UTF-8 text "
                         f"({exc})") from exc
    if not text.strip():
        raise SweepError(f"--long-prompt-file {path_str} holds no text; the long shape "
                         "exists to measure a real prompt, not an empty one")
    return text


def long_prompt_policy(path_str: str, tokens: int) -> str:
    """What a long-shape artifact says about its own prompts: file, size, and n.

    `PROMPT_POLICY` cannot carry either of the two things a long run has to publish —
    it is a fixed string — so a long run replaces it with this one, which names the
    file it read and the token count the engine returned.
    """
    return (f"{SHAPE_LONG} shape: one captured Lloyd turn read verbatim from "
            f"{path_str} ({tokens} tokens by the engine's /tokenize route), sent as "
            "the prompt of every request at every level. One pass per level (n=1) and "
            "no warm-up pass, so no max-minus-min spread is claimed for any row and "
            "the single pass is a cold prefill — reuse engages after one cold pass "
            "(bench-prefix-reuse.py trap 2). Prefix caching is whatever the pinned "
            "argv booted; per-level reuse is the engine's own "
            "prefix_cache_queries_total/prefix_cache_hits_total delta, not an "
            "assertion made here.")


def _flag_value(argv: list[str], name: str):
    """The value of `--name value` / `--name=value`, or True for a bare switch."""
    dashed = f"--{name}"
    for i, tok in enumerate(argv):
        if tok == dashed:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is None or nxt.startswith("--"):
                return True
            return nxt
        if tok.startswith(f"{dashed}="):
            return tok.split("=", 1)[1]
    return None


def engine_argv(port: int, proc_root: str | Path = "/proc") -> dict:
    """The serving process's own argv, found by the `--port` it was given.

    Read from the process rather than from `start-qwen38-flash-next.sh` because the
    script is what a boot WOULD do and `/proc` is what this engine is doing.
    """
    root = Path(proc_root)
    if not root.is_dir():
        raise SweepError(f"proc root {root} is not a directory; pass --proc-root")
    matches = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [t.decode("utf-8", "replace") for t in raw.split(b"\0") if t]
        if not argv:
            continue
        joined = " ".join(argv)
        if "vllm" not in joined and "api_server" not in joined:
            continue
        serving_port = None
        for i, tok in enumerate(argv):
            if tok == "--port" and i + 1 < len(argv):
                serving_port = argv[i + 1]
            elif tok.startswith("--port="):
                serving_port = tok.split("=", 1)[1]
        if serving_port is not None and int(serving_port) == int(port):
            matches.append({"pid": int(entry.name), "argv": argv})
    if not matches:
        raise SweepError(
            f"no vLLM process serving port {port} under {root}; the artifact would "
            "pin a config nobody is running, so the sweep refuses to write")
    if len(matches) > 1:
        raise SweepError(
            f"{len(matches)} processes serve port {port} under {root}; refusing to "
            "guess which one the curve describes")
    return matches[0]


def engine_config(base_url: str, proc_root: str | Path = "/proc") -> dict:
    """Everything the curve is a function of, from the engine's own argv."""
    parsed = urllib.parse.urlparse(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    found = engine_argv(port, proc_root)
    argv = found["argv"]
    flags = {name: _flag_value(argv, name) for name in ENGINE_FLAGS}
    # --speculative-config is a JSON document on the command line; keep it a document
    # in the artifact so a reader can diff two sweeps' spec configs field by field.
    spec = flags.get("speculative-config")
    if isinstance(spec, str) and spec.lstrip().startswith(("{", "[")):
        try:
            flags["speculative-config"] = json.loads(spec)
        except json.JSONDecodeError:
            pass                                  # keep the raw token; it is still a pin
    if flags["max-num-seqs"] in (None, True):
        raise SweepError(
            "the serving process does not carry --max-num-seqs, so the curve would "
            "be unpinnable: its engine default is unknown to this file and a sweep "
            "whose max_num_seqs is unknown is not comparable to another")
    quant = flags["quantization"]
    if quant in (None, True):
        # No --quantization means the checkpoint carries its own format; name where
        # the precision actually comes from instead of writing "unknown".
        model_arg = _flag_value(argv, "model") or ""
        quant = f"not passed as --quantization; from the checkpoint {model_arg!r}"
    return {
        "pid": found["pid"],
        "port": port,
        "max_num_seqs": flags["max-num-seqs"],
        "quantization_or_precision": quant,
        "speculative_config": flags["speculative-config"],
        "flags": flags,
        "argv": argv,
    }


def pool_state(status_url: str, timeout: float = 10.0) -> dict:
    body = json.loads(http_get(status_url, timeout))
    pool = body.get("pool") or {}
    return {
        "url": status_url,
        "initialized": body.get("initialized"),
        "running": pool.get("running"),
        "paused": pool.get("paused"),
        "paused_by": pool.get("paused_by"),
        "in_flight_count": pool.get("in_flight_count"),
    }


def check_pool(status_url: str, declared: str) -> dict:
    """Refuse unless the pool reports the quiet state the operator declared.

    Checked rather than believed in both directions: a declared pause that the
    database does not hold means the fleet is NOT paused, and a "drained" claim
    with jobs in flight means the curve would measure the fleet, not the engine.
    """
    st = pool_state(status_url)
    if st["initialized"] is not True:
        raise SweepError(f"{status_url} did not report an initialized pool "
                         f"({st!r}); a sweep cannot claim a drained pool it cannot read")
    if declared == "paused":
        if st["paused"] is not True:
            raise SweepError(f"--pool paused declared but the pool reports paused="
                             f"{st['paused']!r} (holder {st['paused_by']!r}); POST "
                             "/api/workers/pause first")
    elif st["in_flight_count"] not in (0, None):
        raise SweepError(f"--pool drained declared but the pool reports "
                         f"in_flight_count={st['in_flight_count']}; drain or pause first")
    elif st["in_flight_count"] is None:
        raise SweepError(f"--pool drained declared but {status_url} reports no "
                         "in_flight_count; refusing to assume quiet")
    return st


class Monitor(threading.Thread):
    """Samples in-flight requests so each level can prove it ran alone."""

    def __init__(self, base_url: str, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.base_url = base_url
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                c = counter_sums(metrics_text(self.base_url, 2.0),
                                 ("vllm:num_requests_running", "vllm:num_requests_waiting"))
                self.samples.append((time.time(),
                                     (c.get("vllm:num_requests_running") or 0.0)
                                     + (c.get("vllm:num_requests_waiting") or 0.0)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def peak_between(self, t0: float, t1: float) -> float | None:
        vals = [v for ts, v in self.samples if t0 <= ts <= t1]
        return max(vals) if vals else None


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. At one sample p95 is that sample, which is honest
    for a level of one and is why the table also carries the repeat spread."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(-(-q / 100.0 * len(ordered) // 1)) - 1))
    return ordered[idx]


def _stream_one(base_url: str, model: str, prompt: str, max_tokens: int,
                barrier: "threading.Barrier", timeout: float) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Match live worker traffic: the engine runs --scheduling-policy priority,
        # and a priority-0 sweep preempts real work and forces its prefills to
        # recompute — the contamination would be self-inflicted.
        "priority": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})

    barrier.wait()
    t_start = time.perf_counter()
    ttft: float | None = None
    last = t_start
    completion: int | None = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = b""
            while True:
                piece = resp.read(4096)
                if not piece:
                    break
                buf += piece
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    usage = data.get("usage")
                    if usage:
                        completion = usage.get("completion_tokens", completion)
                    for ch in data.get("choices", []):
                        delta = ch.get("delta", {})
                        if delta.get("content") or delta.get("reasoning") \
                                or delta.get("reasoning_content"):
                            now = time.perf_counter()
                            if ttft is None:
                                ttft = now - t_start
                            last = now
                            chunks += 1
    except Exception as exc:                                    # noqa: BLE001
        return {"error": str(exc)}
    if ttft is None:
        return {"error": "no token received"}
    tokens = completion if completion else chunks
    return {
        "ttft_s": ttft,
        "t_first": t_start + ttft,
        "t_last": last,
        "completion_tokens": tokens,
        "itl_s": ((last - (t_start + ttft)) / (tokens - 1)) if tokens > 1 else None,
    }


def run_level(base_url: str, model: str, level: int, max_tokens: int,
              timeout: float = 300.0, prompt: str | None = None) -> dict:
    barrier = threading.Barrier(level)
    # The long shape puts ONE captured prompt on every slot — that is the prefill being
    # measured. The short shape cycles the eight so prefix caching cannot collapse a
    # level into a single cached stream.
    prompts = ([prompt] * level if prompt is not None
               else [PROMPTS[i % len(PROMPTS)] for i in range(level)])
    with cf.ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(_stream_one, base_url, model, prompts[i], max_tokens,
                               barrier, timeout) for i in range(level)]
        results = [f.result() for f in cf.as_completed(futures)]
    ok = [r for r in results if "error" not in r]
    if not ok:
        raise SweepError(f"level {level}: every request failed "
                         f"({results[0].get('error')})")
    window = max(r["t_last"] for r in ok) - min(r["t_first"] for r in ok)
    total = sum(r["completion_tokens"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok]
    itls = [r["itl_s"] for r in ok if r["itl_s"]]
    return {
        "level": level,
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "decode_tokens": total,
        "decode_window_s": round(window, 3),
        "aggregate_generation_tokens_per_s": round(total / window, 1) if window > 0 else 0.0,
        "ttft_mean_ms": round(statistics.mean(ttfts) * 1000, 1),
        "ttft_p95_ms": round(percentile(ttfts, 95) * 1000, 1),
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 1) if itls else None,
    }


def aggregate_level(level: int, runs: list[dict], shape: str = SHAPE_SHORT) -> dict:
    """Mean AND max-minus-min spread across the repeats of one level.

    Every row carries its `shape`, because the two kinds of row are not comparable and
    one artifact can now hold only ever one of them. A `long_prefill` level has exactly
    one run by construction, and a max-minus-min over one sample is `0.0` — a number
    that reads to the next reader as "the noise was measured, and it is zero". It was
    not measured, so for that shape the spread keys are ABSENT rather than zero-valued:
    absence is the only honest way to say "no claim here" in an artifact that a later
    run will diff key by key.
    """
    out: dict = {"level": level, "shape": shape, "repeats": len(runs), "runs": runs}
    out["n_ok"] = sum(r["n_ok"] for r in runs)
    out["n_err"] = sum(r["n_err"] for r in runs)
    for key in ("aggregate_generation_tokens_per_s", "ttft_mean_ms", "ttft_p95_ms",
                "itl_mean_ms"):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = round(statistics.mean(vals), 1)
            if shape != SHAPE_LONG:
                out[f"{key}_spread"] = round(max(vals) - min(vals), 1)
    return out


def sweep(base_url: str, levels: list[int], repeats: int, max_tokens: int,
          model: str, model_id: str, config: dict, pool_status_url: str,
          declared_pool: str, monitor_interval: float = 0.5,
          warmup: bool = True, long_prompt: str | None = None,
          prompt_policy: str | None = None) -> dict:
    """Run the ladder once, in one shape.

    With `long_prompt` the run is the n=1 long-prefill shape: one pass per level no
    matter what `--repeats` says, no warm-up pass, and the captured prompt on every
    slot. Skipping the warm-up is not an optimisation — reuse engages after one cold
    pass, so warming up would spend the only measured pass on a warm prefill and
    publish it as the cold one.
    """
    shape = SHAPE_LONG if long_prompt is not None else SHAPE_SHORT
    passes = 1 if long_prompt is not None else repeats
    mon = Monitor(base_url, monitor_interval)
    mon.start()
    levels_out: list[dict] = []
    excluded: list[dict] = []
    try:
        if warmup and long_prompt is None:
            run_level(base_url, model, 1, max_tokens)
        for level in levels:
            before = engine_snapshot(base_url)
            t0 = time.time()
            runs = [run_level(base_url, model, level, max_tokens, prompt=long_prompt)
                    for _ in range(passes)]
            t1 = time.time()
            after = engine_snapshot(base_url)
            agg = aggregate_level(level, runs, shape)
            agg["engine_before"] = before
            agg["engine_after"] = after
            agg["prefix_cache_during"] = prefix_delta(before, after)
            pre = after.get("preemptions_total")
            pre0 = before.get("preemptions_total")
            agg["preemptions_during"] = (pre - pre0) if (pre is not None and pre0 is not None) else None
            peak = mon.peak_between(t0, t1)
            agg["observed_inflight_peak"] = peak
            # A peak ABOVE the level launched means something outside the sweep was
            # in the engine: the number describes two workloads sharing a GPU.
            agg["contaminated"] = peak is not None and peak > level
            (excluded if agg["contaminated"] else levels_out).append(agg)
    finally:
        mon.stop()
    return {
        "levels": levels_out,
        "excluded_levels": excluded,
        "requested_levels": levels,
        "repeats": repeats,
        "shape": shape,
        #: What each level actually ran. A long-shape artifact has `repeats` (what was
        #: asked) and this (what ran) disagree by design, and the difference is the
        #: n=1 that makes its rows non-comparable to the short curve.
        "repeats_per_level": passes,
        "max_tokens": max_tokens,
        "base_url": base_url,
        "model": model,
        "model_resolution": resolve_model(base_url, model),
        "prompt_policy": prompt_policy if prompt_policy is not None else PROMPT_POLICY,
        "engine_config": config,
        "pool_declaration": declared_pool,
        "pool_at_write": pool_state(pool_status_url),
        "spec_decode_counts": spec_decode_counts(base_url),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }


def default_out_dir() -> Path:
    """`$LLOYD_DATA/eval`, beside the other eval baselines — never /tmp.

    Through `app.paths` rather than `$LLOYD_DATA` by hand: the data root moved in
    the 2026-09-22 cutover and its resolution has rules a script should not
    re-implement.
    """
    try:
        from app.paths import DATA_ROOT
        return Path(DATA_ROOT) / "eval"
    except Exception:
        return Path.home() / "lloyd-data" / "eval"


def artifact_path(out_dir: str | Path, label: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return Path(out_dir) / f"concurrency-sweep-{stamp}-{label}.json"


def write_artifact(out_dir: str | Path, doc: dict, label: str = "run") -> Path:
    path = artifact_path(out_dir, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_levels(text: str) -> list[int]:
    toks = text.split(",")
    if any(not tok.strip() for tok in toks):
        raise SweepError(f"--levels {text!r} has an empty entry; a dropped number "
                         "would silently sweep a different curve")
    try:
        levels = [int(tok) for tok in toks]
    except ValueError as exc:
        raise SweepError(f"--levels {text!r} is not a comma list of integers") from exc
    if not levels:
        raise SweepError("--levels was empty")
    if any(n < 1 for n in levels):
        raise SweepError(f"--levels {text!r} holds a level below 1")
    if len(set(levels)) != len(levels):
        raise SweepError(f"--levels {text!r} repeats a level; the spread needs distinct ones")
    return levels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", "--users", dest="levels", default="1,2,4,6,8,12,16",
                    help="Comma list of concurrency levels to sweep (--users is the "
                         "old spelling of the same flag)")
    ap.add_argument("--repeats", "--runs", dest="repeats", type=int, default=3,
                    help="Passes per level; the table reports max-minus-min across them")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=None,
                    help="Served model id; default is what /v1/models lists first. "
                         "There is no built-in model name any more — the old one "
                         "named a retired model.")
    ap.add_argument("--pool", choices=("drained", "paused"), default="drained",
                    help="What you assert about the worker pool; verified against "
                         "--pool-status before the first request and again at write")
    ap.add_argument("--pool-status", default=DEFAULT_POOL_STATUS)
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--out", default=None, help="Default $LLOYD_DATA/eval")
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--long-prompt-file", dest="long_prompt_file", default=None,
                    help="Long-prefill shape: send THIS file's contents verbatim as the "
                         "prompt of every request instead of the built-in ~130-token "
                         "prompts. Each level then runs exactly ONCE regardless of "
                         "--repeats (n=1, no warm-up), every row is labelled "
                         f"{SHAPE_LONG}, it carries no max-minus-min spread, and it "
                         "prints in its own table rather than the knee table. The "
                         "prompt's token count is taken from the engine's /tokenize "
                         "route and named in prompt_policy; an engine that will not "
                         "tokenize is a refusal.")
    args = ap.parse_args(argv)

    try:
        levels = parse_levels(args.levels)
        # Before loading the engine: a sweep of a busy pool is not a sweep.
        check_pool(args.pool_status, args.pool)
        # A missing prompt file is refused here, before a single request is loaded.
        long_prompt = (read_long_prompt(args.long_prompt_file)
                       if args.long_prompt_file else None)
        resolved = resolve_model(args.base_url, args.model)
        policy = None
        if long_prompt is not None:
            policy = long_prompt_policy(
                args.long_prompt_file,
                tokenize_count(args.base_url, resolved["model"], long_prompt))
        config = engine_config(args.base_url, args.proc_root)
        doc = sweep(args.base_url, levels, args.repeats, args.max_tokens,
                    resolved["model"], resolved["model"], config, args.pool_status,
                    args.pool, warmup=args.warmup, long_prompt=long_prompt,
                    prompt_policy=policy)
        # Re-checked: a job that started mid-sweep contaminates the artifact, and
        # the operator's declaration describes the window they opened, not this one.
        check_pool(args.pool_status, args.pool)
        out_dir = Path(args.out) if args.out else default_out_dir()
        path = write_artifact(out_dir, doc, args.label)
    except SweepError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    doc["artifact"] = str(path)
    print(f"model={doc['model']} max_num_seqs={config['max_num_seqs']} "
          f"levels={[l['level'] for l in doc['levels']]} repeats={args.repeats}")
    # Two shapes, two tables, and the split is not cosmetic: a long row is n=1, so a
    # reader who found it in the knee table would read a missing spread as a measured
    # zero and a single cold prefill as a point on a curve. The short table's columns
    # and rows are exactly what they were before this flag existed.
    short_rows = [l for l in doc["levels"] if l.get("shape") != SHAPE_LONG]
    long_rows = [l for l in doc["levels"] if l.get("shape") == SHAPE_LONG]
    if short_rows:
        print(f"  {'lvl':>4}  {'agg tok/s':>10} {'spread':>7}  "
              f"{'ttft mean ms':>13} {'spread':>7}  {'ttft p95 ms':>12} {'spread':>7}  "
              f"{'preempt':>8} {'kv':>6}")
        for l in short_rows:
            print(f"  {l['level']:>4}  {l.get('aggregate_generation_tokens_per_s_mean', 0):>10} "
                  f"{l.get('aggregate_generation_tokens_per_s_spread', 0):>7}  "
                  f"{l.get('ttft_mean_ms_mean', 0):>13} {l.get('ttft_mean_ms_spread', 0):>7}  "
                  f"{l.get('ttft_p95_ms_mean', 0):>12} {l.get('ttft_p95_ms_spread', 0):>7}  "
                  f"{str(l['preemptions_during']):>8} "
                  f"{(l['engine_after'].get('kv_cache_usage') or 0):>6.2f}")
    if long_rows:
        print(f"  {SHAPE_LONG.upper()} SHAPE — n=1 per level: one pass of one captured "
              f"prompt, no warm-up. NOT the knee table: with n=1 there is no "
              f"max-minus-min spread across repeats to report, so none is printed, and "
              f"the one pass is a cold prefill (reuse engages after one cold pass).")
        print(f"  {'lvl':>4}  {'agg tok/s':>10}  {'ttft mean ms':>13} {'ttft p95 ms':>12}  "
              f"{'preempt':>8} {'kv':>6}  {'prefix q':>10} {'prefix hits':>12} {'reuse':>7}")
        ratio_printed = False
        for l in long_rows:
            pc = l.get("prefix_cache_during") or {}
            q, h, ratio = pc.get("queries_delta"), pc.get("hits_delta"), pc.get("hit_rate")
            ratio_printed = ratio_printed or ratio is not None
            print(f"  {l['level']:>4}  "
                  f"{l.get('aggregate_generation_tokens_per_s_mean', 0):>10}  "
                  f"{l.get('ttft_mean_ms_mean', 0):>13} {l.get('ttft_p95_ms_mean', 0):>12}  "
                  f"{str(l['preemptions_during']):>8} "
                  f"{(l['engine_after'].get('kv_cache_usage') or 0):>6.2f}  "
                  f"{('--' if q is None else f'{q:,.0f}'):>10} "
                  f"{('--' if h is None else f'{h:,.0f}'):>12} "
                  f"{('--' if ratio is None else f'{ratio * 100:.1f}%*'):>7}")
        if ratio_printed:
            # The column is meaningless without its condition, so it never travels
            # alone: see bench-prefix-reuse.py:11-25 on the 11.2x MTP case where the
            # ratio read ~68% against zero real cross-request reuse.
            print(f"  * reuse column: {PREFIX_RATE_CONDITION}")
    for l in doc["excluded_levels"]:
        print(f"  EXCLUDED level {l['level']}: in-flight peak {l['observed_inflight_peak']} "
              f"against a level of {l['level']} — other traffic shared the engine")
    print(f"artifact: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
