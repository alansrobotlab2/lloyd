"""The djev determinism probe (#1357).

Two behaviours matter and both are graded here against a scripted engine, because
neither can be trusted to a live GPU:

* A boot that is not deterministic must **fail** (`exit 1`). This is the check
  the kernel bisect is run against; a probe that prints numbers and returns 0
  regardless would let a variant be recorded as passing.
* A host with no djev on it must **pass** (`exit 0`) with an explicit
  `engine unreachable` line. GPU 2 is single-tenant (`start-djev.sh` header):
  the Qwen3.6 secondary owns it on some boots, and `:8011/health` is the signal
  production itself uses for that (`app/supervisor_client.py`,
  `scripts/service_health_check.py`). A probe that returned failure there would
  report a healthy machine as broken, which is the same class of defect as the
  one #1357 is bisecting.

The scripted engine also lets the tests pin the thing a real one cannot: that the
replayed requests are actually byte-identical, and that the prefix-cache
counters the probe prints are read from the engine rather than assumed.
"""

from __future__ import annotations

import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.djev_determinism_probe import (
    HEADER_ROW_PLACEHOLDER, HITS_TOTAL, QUERIES_TOTAL, default_prompt, main, max_abs_delta,
    parse_counters)
# The row the probe now emits is graded by the same expression that grades the
# script header it is pasted into, so the two cannot drift apart.
from tests.test_start_djev_flags import VARIANT_RE


class FakeEngine:
    """Answers /health, /metrics and /v1/completions from a scripted top-k.

    `script` is the list of top_logprobs dicts returned per completion, cycled
    once exhausted. Counters move with every completion the way a real engine's
    do: `queries` by the prompt length, and `hits` only by the part of the prompt
    the engine had ALREADY seen — the request that first carries a prompt is the
    one that populates the cache, so it reports a miss (live engine, warm regime:
    run 1 `hits+0`, run 2 `hits+4768` of 4800). A fake that credited every
    unsalted request would let the tests assert a warm baseline that no engine
    produces, which is the confound this instrument exists to remove.
    """

    # A prefix hit is whole reusable blocks only, and the block holding the
    # position being generated cannot be reused: `((tokens - 1) // 32) * 32`.
    # That reproduces both numbers the live engine printed — 4800 tokens ->
    # hits+4768 and 180 tokens -> hits+160 — which is why CHUNK is 32 and not a
    # guess: the fixture has to make the boundary the engine makes, or a test can
    # assert a cache delta no boot produces.
    CHUNK = 32

    def __init__(self, script, *, no_logprobs=False):
        self.script = list(script)
        self.no_logprobs = no_logprobs
        self.calls = 0
        self.bodies: list[dict] = []
        self.queries = 0.0
        self.hits = 0.0
        self.seen_prompts: set[tuple[str, str | None]] = set()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code, payload, ctype="application/json"):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, b"ok", "text/plain")
                elif self.path == "/metrics":
                    body = "".join(
                        f'{name}{{model_name="djev",engine_index="0"}} {value:.1f}\n'
                        for name, value in ((QUERIES_TOTAL, outer.queries),
                                            (HITS_TOTAL, outer.hits)))
                    self._send(200, body.encode(), "text/plain")
                else:
                    self._send(404, b"nope", "text/plain")

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path != "/v1/completions":
                    self._send(404, {"error": "nope"})
                    return
                outer.bodies.append(body)
                salt = body.get("cache_salt")
                key = (body.get("prompt", ""), "" if salt is None else str(salt))
                # ~4 chars per token is what the live tokenizer measured for this
                # prompt (19,040 chars -> 4,800 tokens); the fixture only needs to
                # scale with the prompt so a shape test can read the number back.
                tokens = max(1, len(body.get("prompt", "")) // 4)
                reusable = ((tokens - 1) // outer.CHUNK) * outer.CHUNK
                outer.queries += tokens
                if key in outer.seen_prompts:
                    outer.hits += reusable
                else:
                    # The request that first carries this prompt+salt populates
                    # the cache; it gets no hits, exactly as on the live engine
                    # (warm run 1 -> hits+0, run 2 -> hits+4768).
                    outer.seen_prompts.add(key)
                top = outer.script[outer.calls % len(outer.script)]
                outer.calls += 1
                choice: dict = {"index": 0, "text": "\n", "finish_reason": "length"}
                if not outer.no_logprobs:
                    choice["logprobs"] = {"tokens": ["\n"], "top_logprobs": [top]}
                self._send(200, {
                    "id": "cmpl-x", "model": "djev", "object": "text_completion",
                    "choices": [choice],
                    "usage": {"prompt_tokens": tokens,
                              "completion_tokens": 1, "total_tokens": tokens + 1},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _closed_port() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


#: 8704 chars is 2176 tokens at the fixture's 4 chars/token, and 2176 is the
#: prompt length #1357 measured, whose reusable part is 2144 blocks-aligned
#: tokens. So the counters this fake moves are the ones the live engine printed,
#: not ones chosen to make an assertion pass.
FIXED_PROMPT = ("determinism probe fixed prompt " * 400)[:8704]


def _run(engine_url, structured_url, *extra, prompt: str | None = FIXED_PROMPT) -> tuple[int, str]:
    """Run the probe's own main() against `engine_url`, capturing its report."""
    import contextlib
    import io
    buf = io.StringIO()
    argv = ["--engine-url", engine_url, "--structured-url", structured_url,
            "--timeout", "5", *list(extra)]
    if prompt is not None:
        argv += ["--prompt", prompt]
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


STABLE = [{" the": -0.1, " a": -6.9, " no": -8.1}]
DRIFTING = [{" the": -0.1, " a": -6.96}, {" the": -0.1, " a": -5.67}]


# ── Clause 2: it measures, prints, and fails when the labels move ────────────

def test_a_stable_argmax_with_moving_labels_still_exits_nonzero(capsys):
    """#1357's exact signature: top-1 identical on every run, the secondary
    labels moving by nats. A check on the argmax alone would pass this boot."""
    with FakeEngine(DRIFTING) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3")
    assert rc == 1, f"a boot that disagrees must not pass: rc={rc}\n{out}"
    # 1.29 nats is exactly |-6.96 - -5.67| from the item's own signature.
    assert "max |delta label logprob| = 1.2900 nats" in out, out
    assert "NOT DETERMINISTIC" in out


def test_identical_label_logprobs_pass_in_both_cache_regimes(capsys):
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "5")
    assert rc == 0, out
    assert "max |delta label logprob| = 0.0000 nats" in out, out
    assert "warm" in out and "cold" in out
    assert "DETERMINISTIC" in out


def test_it_prints_the_prefix_cache_delta_for_every_request():
    """The cache state is the confound: 'the kernels vary' is only admissible
    while the counters say the cache did not.

    The sequence asserted below is the one the live engine printed, not the one
    that is convenient to check. Warm, run 1 is the request that POPULATES the
    cache, so it reports `hits+0` even unsalted; only the runs after it report
    the reused block (live: run 1 `hits+0`, run 2 `hits+4768` of 4800 queries).
    A fake that credited every unsalted request would let this file certify a
    warm baseline of constant hits that no boot produces — and the probe's own
    live output already contradicts that, so the assertion here is written to
    the contradiction rather than to the guess.
    """
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3", "--regime", "warm")
    assert rc == 0, out
    warm = [line for line in out.splitlines() if line.startswith("warm  run ")]
    assert len(warm) == 3, out
    assert "queries+2176 hits+0" in warm[0], \
        f"the first warm request populates the cache and gets no hits: {warm[0]}"
    for line in warm[1:]:
        assert "queries+2176 hits+2144" in line, \
            f"a repeated warm request reuses the aligned block: {line}"
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2", "--regime", "cold")
    assert rc == 0, out
    assert out.count("queries+2176 hits+0") == 2, \
        "a fresh salt per request must never report a hit, or the cold regime is warm"


def test_every_run_sends_the_same_bytes_apart_from_the_cold_salt():
    """If the prompt drifted, the probe would be measuring prompt length and
    calling it kernels — which is the trap #1357's triage named for whoever
    picks this up."""
    with FakeEngine(STABLE) as eng:
        _run(eng.url, eng.url, "--runs", "2", "--regime", "cold")
    assert len(eng.bodies) == 2
    prompts = {b["prompt"] for b in eng.bodies}
    assert len(prompts) == 1, "the probe varied its own input between runs"
    assert all(b["max_tokens"] == 1 and b["logprobs"] == 5 for b in eng.bodies)
    assert len({b["cache_salt"] for b in eng.bodies}) == 2, \
        "the cold regime must salt every request, or the prefix cache is being measured instead"


def test_the_built_in_prompt_stays_past_the_2048_token_compile_range():
    """Below that endpoint the engine runs a different compiled range, which is a
    different regime rather than the one #1357 measured: the same engine returned
    four byte-identical answers on the short structured prompt and nats of drift
    on the ~4.8k-token completions prompt. So the prompt the probe ships with has
    to stay past it, and the only way to know that is to ask the engine how long
    the prompt it received was — which is what `prompt_tokens=` in the report is
    for. Asserted through main() with no --prompt, so it checks the built-in
    default, not a test-local string."""
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2", "--regime", "warm",
                       prompt=None)
    assert rc == 0, out
    seen = {int(m.group(1)) for m in re.finditer(r"prompt_tokens=(\d+)", out)}
    assert len(seen) == 1, f"the probe varied its own prompt length: {seen}"
    tokens = seen.pop()
    assert tokens > 2048, (
        f"the built-in default prompt is {tokens} tokens, inside the 2048 "
        f"compile_ranges_endpoints the production prompt is past")
    assert tokens == len(default_prompt()) // 4, \
        "the fixture counts 4 chars/token; the report must be the built-in prompt's length"


def test_a_reordered_topk_fails_even_when_the_values_are_identical():
    """Positional subtraction already catches a re-order that carries different
    values; what it cannot see is the same two values swapped. That is the case
    the changed flag exists for, and #1357's signature includes it: the top-5
    set changed run to run."""
    delta, changed = max_abs_delta([("a", -1.0), ("b", -1.0)],
                                   [("b", -1.0), ("a", -1.0)])
    assert delta == 0.0 and changed, "a re-ordered top-k is a difference the caller must see"
    with FakeEngine([{" a": -1.0, " b": -1.0}, {" b": -1.0, " a": -1.0}]) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2", "--regime", "warm")
    assert rc == 1, f"a top-k that changes run to run is not deterministic: {out}"
    assert "max |delta label logprob| = 0.0000 nats" in out, out
    assert "the returned top-k itself changed run to run" in out


def test_one_run_is_refused_rather_than_reporting_a_vacuous_zero():
    """A check with no pair to compare reports a verdict it cannot justify."""
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "1")
    assert rc == 2 and "at least 2" in out, out


def test_an_answer_with_no_logprobs_is_a_probe_error_not_a_verdict():
    """A build that drops the logprobs field answers every request identically:
    an empty dict equal to the next one. Reporting that as deterministic would
    certify any engine that cannot be scored."""
    with FakeEngine(STABLE, no_logprobs=True) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2")
    assert rc == 2, f"a probe that cannot read the labels has no verdict: rc={rc}\n{out}"
    assert "probe error" in out
    assert "DETERMINISTIC" not in out


def test_the_worst_regime_is_the_one_the_row_reports():
    with FakeEngine(DRIFTING) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3", "--regime", "cold",
                       "--variant-label", "variant-under-test")
    assert rc == 1, out
    row = [line for line in out.splitlines() if line.startswith("row: ")][0]
    assert "variant-under-test" in row and "cold 1.2900" in row, row


# ── Clause 2 (#1363): a row the sweep can paste without hand-formatting ──────

def _header_rows(out: str) -> list[str]:
    """The lines shaped like a `start-djev.sh` header row, placeholder and all."""
    return [ln for ln in out.splitlines() if ln.startswith("#   ")]


def _measured(out: str, regime: str) -> str:
    m = re.search(rf"^{regime}\s+max \|delta label logprob\| = (\d+\.\d+) nats",
                  out, re.MULTILINE)
    assert m, f"no {regime} summary line in:\n{out}"
    return m.group(1)


def test_the_probe_prints_a_row_the_script_header_would_accept():
    """#1361's attended window transcribes each trial into start-djev.sh's
    header table by hand, and that table is graded by VARIANT_RE, which is
    unforgiving: three spaces after the `#`, and the trailing `/ 0` or `/ 1` is
    mandatory because it is what stops a prose line carrying three numbers from
    being read as a measurement. The probe printed a human-readable `row:`; the
    window needs the row itself, with the 81-query p50 the only number left."""
    with FakeEngine(DRIFTING) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3",
                       "--variant-label", "triton", "--batch-invariant", "0")
    assert rc == 1, out
    rows = _header_rows(out)
    assert len(rows) == 1, f"exactly one header-ready row expected, got {rows}"
    row = rows[0]
    assert VARIANT_RE.match(row) is None, (
        f"a row whose p50 is a placeholder must not read as measured: {row!r}")
    filled = row.replace(HEADER_ROW_PLACEHOLDER, "510.5", 1)
    m = VARIANT_RE.match(filled)
    assert m, f"substituting a real ms number must yield a valid header row: {filled!r}"
    # VARIANT_RE's `variant` group runs through the lever pair, so it captures
    # `triton / 0`; `inv` is the nested BATCH_INVARIANT digit (#1357's row shape).
    assert m["variant"] == "triton / 0" and m["inv"] == "0", m.groups()
    assert m["p50"] == "510.5", m.groups()


def test_the_header_row_carries_the_numbers_this_run_measured():
    """A row printing a plausible number the probe never measured is worse than
    no row at all: it lands in the header table as a trial that happened. Both
    decimals are read back out of the per-regime summary lines printed in the
    same run, and the one slot that is genuinely unknown stays non-numeric."""
    with FakeEngine(DRIFTING) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3", "--variant-label", "triton")
    assert rc == 1, out
    row = _header_rows(out)[0]
    assert row.count(HEADER_ROW_PLACEHOLDER) == 1, row
    assert not any(c.isdigit() for c in HEADER_ROW_PLACEHOLDER), \
        "a numeric placeholder would pass for a measured p50"
    m = VARIANT_RE.match(row.replace(HEADER_ROW_PLACEHOLDER, "510.5", 1))
    assert m, row
    assert m["warm"] == _measured(out, "warm"), f"row warm must be the measured warm: {row}"
    assert m["cold"] == _measured(out, "cold"), f"row cold must be the measured cold: {row}"
    assert m["inv"] == "0", f"BATCH_INVARIANT defaults to 0, the shipped value: {row}"


def test_a_deterministic_boot_prints_the_row_too_because_that_is_the_winner():
    """The row that matters most is the one a variant earned: exit 0 with
    0.0000 in both regimes is the pair the defaults get flipped to, and it must
    reach the header table in the same one-edit form."""
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2",
                       "--variant-label", "marlin", "--batch-invariant", "1")
    assert rc == 0, out
    row = _header_rows(out)[0]
    m = VARIANT_RE.match(row.replace(HEADER_ROW_PLACEHOLDER, "500", 1))
    assert m, row
    assert m["variant"] == "marlin / 1" and m["inv"] == "1", m.groups()
    assert m["cold"] == "0.0000" and m["warm"] == "0.0000", row


def test_a_one_regime_run_emits_no_row_it_cannot_complete():
    """`--regime warm` measures one of the two numbers the header row needs.
    Half a row, pasted, is a fabricated cold measurement, so the probe names
    what is missing instead of printing a line with a slot in it."""
    for regime in ("warm", "cold"):
        with FakeEngine(DRIFTING) as eng:
            rc, out = _run(eng.url, eng.url, "--runs", "2", "--regime", regime)
        assert rc == 1, out
        assert not _header_rows(out), f"--regime {regime} has no cold+warm pair:\n{out}"
        assert "both regimes" in out, f"--regime {regime} should say why: {out}"


# ── Clause 3: no engine is not a failure ────────────────────────────────────

def test_no_engine_answering_is_reported_as_unreachable_and_exits_zero(capsys):
    gone = _closed_port()
    rc, out = _run(gone, gone)
    assert rc == 0, f"a host with no djev must not read as djev failing: rc={rc}"
    assert "engine unreachable" in out, out


def test_the_structured_port_being_down_counts_as_no_engine(capsys):
    """:8011 is production's liveness signal for the whole engine. If it is not
    answering, GPU 2 is somebody else's and there is nothing to bisect."""
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, _closed_port())
    assert rc == 0, f"rc={rc}\n{out}"
    assert "engine unreachable" in out


def test_an_engine_exporting_no_cache_counters_counts_as_unreachable():
    """Without the counters the probe cannot tell a cache effect from a kernel
    one, which is the one thing this instrument exists to separate. It degrades
    like an absent engine rather than printing a number it cannot attribute."""
    class NoMetrics(FakeEngine):
        pass

    with FakeEngine(STABLE) as eng:
        original_get = eng.server.RequestHandlerClass.do_GET

        def do_GET(self):
            if self.path == "/metrics":
                self._send(404, b"not exported", "text/plain")
            else:
                original_get(self)
        eng.server.RequestHandlerClass.do_GET = do_GET
        rc, out = _run(eng.url, eng.url, "--runs", "2")
    assert rc == 0, f"rc={rc}\n{out}"
    assert "engine unreachable" in out, out


def test_the_counter_parsing_sums_labelled_series():
    """vLLM exports these with labels; reading only an unlabelled line would
    report 0 queries and 0 hits and look exactly like a constant cache."""
    text = ('# TYPE vllm:prefix_cache_queries_total counter\n'
            'vllm:prefix_cache_queries_total{model_name="djev",engine_index="0"} 4352.0\n'
            'vllm:prefix_cache_hits_total{model_name="djev",engine_index="0"} 4288.0\n')
    got = parse_counters(text)
    assert got[QUERIES_TOTAL] == 4352.0
    assert got[HITS_TOTAL] == 4288.0
