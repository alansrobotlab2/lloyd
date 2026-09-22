#!/usr/bin/env python3
"""Does djev answer the same request twice? (#1357)

    .venvs/lloyd/bin/python -m scripts.djev_determinism_probe
    .venvs/lloyd/bin/python -m scripts.djev_determinism_probe --runs 5 --variant-label "triton MoE + INV=1"

djev answers a byte-identical request with different label logprobs. Measured
2026-09-21 on a 32-row recall rank replayed straight at vLLM :8010: the argmax
at every canvas position was identical on every run while the label logprobs the
rank score is built from moved by 1-3 nats between identical requests. The
prefix cache was ruled out with the engine's own counters, not by assumption (a
fresh ``cache_salt`` per request varies as much, and the
``vllm:prefix_cache_{hits,queries}_total`` delta was constant while the labels
still moved), and so were CUDA graphs, compile and async scheduling. What is
left is the kernels -- the Marlin NvFp4 MoE backend and TRITON_ATTN -- and the
only way to find out which one is ``agent-services/bin/start-djev.sh``'s
``MOE_BACKEND`` / ``BATCH_INVARIANT`` levers plus this probe.

Why the probe is built the way it is, and what a bisect would otherwise measure
by mistake:

* **It pins the prompt.** The variance is shape-dependent: a prompt crossing the
  2048-token ``compile_ranges_endpoints`` behaves differently from one under it,
  so a probe that regenerated its input per run would measure prompt length and
  call it kernels. The default prompt is one fixed string, sent verbatim, and
  the measured ``prompt_tokens`` is printed on every line so two sessions can
  tell whether they compared the same shape.
* **It runs both cache regimes.** Warm (the prefix cache warm on the same
  prompt) and cold (a fresh ``cache_salt`` per request, which the engine's
  counters confirm at ``hits+0``). They have different signatures -- warm moved
  2.91 nats, cold 11.07 nats and flipped the top-1 token -- and a config that
  passes one while failing the other has not been made deterministic, only
  hidden. A passing boot prints 0.0000 in both.
* **It prints the cache delta per request** rather than trusting it. "The
  kernels vary" is only admissible while the cache state is measured constant,
  so every run carries its own ``queries+`` / ``hits+`` evidence. Read it as the
  engine writes it: the first warm request is the one that fills the cache, so it
  reports ``hits+0`` too, and the constancy the warm regime rests on is the one
  its runs 2..N share (live, 4800 tokens: run 1 ``hits+0``, then ``hits+4768`` on
  every run -- 4768 because only whole 32-token blocks before the generated
  position are reusable, ``floor((4800-1)/32)*32``). A warm baseline of hits on
  run 1 describes no boot, so nothing here asserts one.
* **It emits the header row in the form the script keeps them in.** A trial's
  numbers land in ``start-djev.sh``'s variant table, which
  ``tests/test_start_djev_flags.py::VARIANT_RE`` grades for shape -- three spaces
  after the ``#``, and a trailing ``/ 0`` or ``/ 1`` that is mandatory because it
  is what stops a prose line carrying three numbers from being read as a
  measurement. Hand-transcribing that from a prose line is how a row goes wrong
  (#1363), so the probe prints the row itself with only its p50 left as
  ``PASTE_MS``: the one number it genuinely cannot know, in the one slot the
  window has to fill.
* **It exits 0, not 1, when the engine is not there.** GPU 2 is single-tenant
  (`start-djev.sh` header): on a host where the Qwen secondary holds the card
  there is no djev to fail on, and a probe that returns failure there turns an
  unrelated machine into a broken build. That branch prints an explicit
  ``engine unreachable`` line and returns 0; a real disagreement returns 1.

Exit codes: 0 deterministic (or no engine to ask), 1 byte-identical requests
disagreed, 2 the engine answered but not in the shape expected.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.error
import uuid
import urllib.request
from typing import Any, Sequence

#: The raw vLLM engine: where the label logprobs actually live. The structured
#: server on :8011 folds them into per-question scores, so the thing under test
#: has to be read one level below it.
DEFAULT_ENGINE_URL = "http://127.0.0.1:8010"
#: The structured server. :8011/health is production's liveness signal for the
#: whole engine (`app/supervisor_client.py`, `scripts/service_health_check.py`),
#: so asking it is how the probe knows it is on a host with no djev rather than
#: on one where djev is merely disagreeing with itself.
DEFAULT_STRUCTURED_URL = "http://127.0.0.1:8011"

DEFAULT_MODEL = "djev"
DEFAULT_RUNS = 5
#: Same number the item's own probe used. It is also what the rank score is
#: built from: `structured_server` reads the label candidates out of the top-k,
#: so a disagreement outside the top-k cannot move a rank.
DEFAULT_LOGPROBS = 5
DEFAULT_TIMEOUT_S = 30.0
REGIMES = ("warm", "cold")

QUERIES_TOTAL = "vllm:prefix_cache_queries_total"
HITS_TOTAL = "vllm:prefix_cache_hits_total"

#: The one slot in a header row this probe cannot know: the 81-query recall p50
#: comes from `eval/run_eval.py`, in the same attended window but a different
#: instrument (#1361's step 3). It is deliberately non-numeric — the row is
#: graded by `tests/test_start_djev_flags.py::VARIANT_RE`, whose p50 group is
#: `\d+(\.\d+)?`, so a placeholder row cannot be pasted and mistaken for a
#: measurement. Substituting a real ms number for this is what makes the line a
#: row; that round-trip is pinned in tests/test_djev_determinism_probe.py.
HEADER_ROW_PLACEHOLDER = "PASTE_MS"

#: The line `--runs` must be at least this large for the acceptance check to
#: mean anything ("identical across 5 runs"). Below two there is no pair to
#: compare and a 0.0000 would be vacuous -- the class of bug where a check whose
#: denominator can be zero reports a verdict it cannot justify.
MIN_RUNS = 2

#: The fixed prompt, one line repeated. Non-degenerate text (a purely repeated
#: token compresses into a shape no real read has), long enough to sit above the
#: 2048-token compile-range endpoint the triage named, and byte-stable so the
#: only variable left between runs is the engine.
PROMPT_LINES = 160


class TransportError(Exception):
    """Nothing answered. On this box that usually means another tenant has GPU 2."""


class ProbeError(Exception):
    """Something answered, but not in a shape this probe can score."""


def default_prompt(lines: int = PROMPT_LINES) -> str:
    """The byte-identical text every run sends. Deterministic by construction."""
    return "".join(
        f"determinism probe line {i:04d}: the quick brown fox jumps over the "
        f"lazy dog, and the kernel reads the same canvas twice.\n"
        for i in range(lines)
    )


def _http(base: str, path: str, *, body: dict | None = None,
          timeout: float = DEFAULT_TIMEOUT_S) -> tuple[int, bytes]:
    """One HTTP call. Connect-level failure -> TransportError; status -> returned."""
    url = base.rstrip("/") + path
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:  # a real answer with a bad status
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise TransportError(f"{url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        raise TransportError(f"{url}: {exc}") from exc


def check_engine(engine_url: str, structured_url: str, *,
                 timeout: float = DEFAULT_TIMEOUT_S) -> str | None:
    """None when both ports answer; else the reason, for the `engine unreachable` line."""
    for base in (engine_url, structured_url):
        try:
            status, _ = _http(base, "/health", timeout=timeout)
        except TransportError as exc:
            return str(exc)
        if status >= 400:
            return f"{base}/health: HTTP {status}"
    return None


def parse_counters(text: str) -> dict[str, float]:
    """Sum the Prometheus exposition for every label series of each counter.

    vLLM exports these with labels (`{model_name="djev",engine_index="0"}`), and
    a probe that read only an unlabeled line would see 0 hits and 0 queries and
    conclude the cache was constant. That is the same failure this whole item is
    about, one level up.
    """
    wanted = (QUERIES_TOTAL, HITS_TOTAL)
    out = {name: 0.0 for name in wanted}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        name = key.split("{", 1)[0]
        if name in wanted:
            try:
                out[name] += float(value)
            except ValueError:
                continue
    return out


def metrics_unusable(text: str, status: int) -> str | None:
    """Why the exposition cannot carry the cache counters, or None if it can.

    A `/metrics` that answers 404 (stats not enabled on the boot) parses to 0
    queries and 0 hits, which is exactly what a constant cache looks like. The
    probe would then print `hits+0` as evidence the cache did not move while
    measuring nothing, and hand back a kernel verdict it cannot attribute — so
    this is refused as unreachable, not treated as a zero.
    """
    if status >= 400:
        return f"HTTP {status}"
    for line in text.splitlines():
        if line and not line.startswith("#"):
            if line.split("{", 1)[0].split(" ", 1)[0] in (QUERIES_TOTAL, HITS_TOTAL):
                return None
    return f"neither {QUERIES_TOTAL} nor {HITS_TOTAL} is exported"


def counters(engine_url: str, *, timeout: float) -> dict[str, float]:
    """The prefix-cache counters, or TransportError when they cannot be read."""
    status, raw = _http(engine_url, "/metrics", timeout=timeout)
    text = raw.decode("utf-8", "replace")
    reason = metrics_unusable(text, status)
    if reason:
        raise TransportError(f"{engine_url}/metrics: {reason}")
    return parse_counters(text)


def read_labels(engine_url: str, *, model: str, prompt: str, logprobs: int,
                cache_salt: str | None = None,
                timeout: float = DEFAULT_TIMEOUT_S) -> tuple[list[tuple[str, float]], int]:
    """The top-k logprobs at the one generated position, in the order returned.

    No `temperature` and no `seed`: this build refuses both for diffusion
    models with an HTTP 400, and the read-only path draws nothing anyway.
    """
    body: dict[str, Any] = {"model": model, "prompt": prompt,
                            "max_tokens": 1, "logprobs": logprobs}
    if cache_salt:
        body["cache_salt"] = cache_salt
    status, raw = _http(engine_url, "/v1/completions", body=body, timeout=timeout)
    if status >= 400:
        raise ProbeError(f"POST /v1/completions: HTTP {status} {raw[:200]!r}")
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        top = choice["logprobs"]["top_logprobs"][0]
        prompt_tokens = int(data["usage"]["prompt_tokens"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProbeError(f"engine answered without logprobs: {exc}") from exc
    # json preserves the object's order, which is vLLM's rank order. Keeping it
    # is what lets a top-k *set* change show up as a delta and not as a tie.
    return list(top.items()), prompt_tokens


def max_abs_delta(first: Sequence[tuple[str, float]],
                  second: Sequence[tuple[str, float]]) -> tuple[float, bool]:
    """(max |delta| nats, did the ordered top-k differ) between two runs.

    Two comparisons, and the headline is the worse of them. By rank position is
    always defined and catches a value that moved; by token identity catches a
    value that moved without changing order, and keeps a re-ordered top-k from
    reading as a coincidence of positions.
    """
    by_rank = 0.0
    for (_, a), (_, b) in zip(first, second):
        by_rank = max(by_rank, abs(a - b))
    lookup_a = dict(first)
    by_token = 0.0
    for token, b in second:
        if token in lookup_a:
            by_token = max(by_token, abs(lookup_a[token] - b))
    set_changed = [t for t, _ in first] != [t for t, _ in second]
    return max(by_rank, by_token), set_changed


def _fmt(labels: Sequence[tuple[str, float]]) -> str:
    return " ".join(f"{t!r}:{lp:+.4f}" for t, lp in labels[:3])


def run_regime(regime: str, engine_url: str, *, model: str, prompt: str, runs: int,
               logprobs: int, timeout: float, out=print) -> tuple[float, bool]:
    """Replay `runs` byte-identical reads in one cache regime.

    Returns (max |delta| nats against the first run, whether any top-k changed).
    """
    baseline: list[tuple[str, float]] | None = None
    worst = 0.0
    changed = False
    for i in range(runs):
        salt = None
        if regime == "cold":
            # A salt the engine has never seen, so the prefix cache cannot be
            # what makes run 2 look like run 1. The counters printed below are
            # the proof it worked (hits+0), not this string.
            salt = f"determinism-probe-{uuid.uuid4().hex}"
        before = counters(engine_url, timeout=timeout)
        labels, prompt_tokens = read_labels(engine_url, model=model, prompt=prompt,
                                            logprobs=logprobs, cache_salt=salt,
                                            timeout=timeout)
        after = counters(engine_url, timeout=timeout)
        dq = after[QUERIES_TOTAL] - before[QUERIES_TOTAL]
        dh = after[HITS_TOTAL] - before[HITS_TOTAL]
        if baseline is None:
            delta_note = "first run"
        else:
            delta, moved = max_abs_delta(baseline, labels)
            worst = max(worst, delta)
            changed = changed or moved
            delta_note = f"max|Δ|={delta:.4f} nats"
        out(f"{regime:<5} run {i + 1}: prompt_tokens={prompt_tokens} "
            f"queries+{dq:.0f} hits+{dh:.0f} {_fmt(labels)} {delta_note}")
        if baseline is None:
            baseline = labels
    return worst, changed


def header_row(variant: str, batch_invariant: str, cold: float, warm: float,
               p50: str = HEADER_ROW_PLACEHOLDER) -> str:
    """One row of `start-djev.sh`'s variant table, in the shape that file's
    pinning test accepts it.

    The spacing is not cosmetic: `tests/test_start_djev_flags.py::VARIANT_RE`
    wants three spaces after the `#`, a `<variant> / <0|1>` pair, then cold,
    warm, p50 -- and the `/ 0` or `/ 1` suffix is load-bearing, because a line
    with three numbers and no lever pair would otherwise be read as a
    measurement. `p50` defaults to the non-numeric placeholder, so what this
    returns is a row that cannot pass for measured data until someone substitutes
    the recall number (#1361's step 3) for it."""
    return (f"#   {variant} / {batch_invariant}    {cold:.4f}    {warm:.4f}    {p50}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    ap.add_argument("--structured-url", default=DEFAULT_STRUCTURED_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--logprobs", type=int, default=DEFAULT_LOGPROBS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--regime", choices=REGIMES + ("both",), default="both")
    ap.add_argument("--prompt", default=None,
                    help="Send this instead of the built-in fixed prompt.")
    ap.add_argument("--prompt-lines", type=int, default=PROMPT_LINES,
                    help="Lines of the built-in fixed prompt (sizes the shape).")
    ap.add_argument("--variant-label", default="incumbent",
                    help="Name of the boot under test; appears in the header row.")
    ap.add_argument("--batch-invariant", choices=("0", "1"), default="0",
                    help="BATCH_INVARIANT the engine was booted with; fills the '/ 0' or "
                         "'/ 1' slot of the header row, which is mandatory and is what "
                         "distinguishes the row from a prose line. The probe reads the "
                         "engine over HTTP and cannot see its argv, so this is the "
                         "boot's own record of that lever — pass 1 for the "
                         "BATCH_INVARIANT=1 trial and leave it out otherwise.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.runs < MIN_RUNS:
        print(f"probe error: --runs must be at least {MIN_RUNS}; "
              f"one request has no pair to compare")
        return 2

    prompt = args.prompt or default_prompt(args.prompt_lines)
    regimes = REGIMES if args.regime == "both" else (args.regime,)

    reason = check_engine(args.engine_url, args.structured_url, timeout=args.timeout)
    if reason:
        # Exit 0: no djev here is not djev failing. See the module docstring.
        print(f"engine unreachable: {reason}")
        return 0

    print(f"djev determinism probe: {args.runs} runs x "
          f"{' + '.join(regimes)} regime(s), logprobs={args.logprobs}, "
          f"engine={args.engine_url}")
    worst_overall = 0.0
    any_changed = False
    per_regime: dict[str, float] = {}
    try:
        for regime in regimes:
            worst, changed = run_regime(
                regime, args.engine_url, model=args.model, prompt=prompt,
                runs=args.runs, logprobs=args.logprobs, timeout=args.timeout)
            per_regime[regime] = worst
            worst_overall = max(worst_overall, worst)
            any_changed = any_changed or changed
            print(f"{regime:<5} max |delta label logprob| = {worst:.4f} nats "
                  f"over {args.runs} runs "
                  f"(top-{args.logprobs} set changed: {'yes' if changed else 'no'})")
    except TransportError as exc:
        print(f"engine unreachable: {exc}")
        return 0
    except ProbeError as exc:
        print(f"probe error: {exc}")
        return 2

    print(f"max |delta label logprob| = {worst_overall:.4f} nats")
    print(f"row: {args.variant_label} | "
          + " | ".join(f"{r} {per_regime[r]:.4f}" for r in regimes)
          + " | (paste the 81-query recall p50 here)")
    cold, warm = per_regime.get("cold"), per_regime.get("warm")
    if cold is not None and warm is not None:
        print(f"header row for start-djev.sh (BATCH_INVARIANT={args.batch_invariant} as "
              f"passed; {HEADER_ROW_PLACEHOLDER} is the only number left to fill in):")
        print(header_row(args.variant_label, args.batch_invariant, cold, warm))
    else:
        print(f"header row: not emitted — the script's table wants cold AND warm and "
              f"--regime {args.regime} measured one. Run the default of both regimes.")
    reasons = []
    if worst_overall > 0.0:
        reasons.append(f"byte-identical requests disagree by {worst_overall:.4f} nats")
    if any_changed:
        reasons.append("the returned top-k itself changed run to run")
    if reasons:
        print(f"NOT DETERMINISTIC: {'; '.join(reasons)}")
        return 1
    print("DETERMINISTIC: identical label logprobs across every run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
