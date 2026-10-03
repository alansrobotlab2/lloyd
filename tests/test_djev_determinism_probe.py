"""The djev determinism probe (#1357, #2116).

Three behaviours matter and all are graded here against a scripted engine,
because none can be trusted to a live GPU:

* A boot whose **seeded** (production-shaped) read is not deterministic must
  **fail** (`exit 1`). This is the check the kernel bisect is run against; a
  probe that prints numbers and returns 0 regardless would let a variant be
  recorded as passing.
* The **unseeded** read — the same prompt with no `vllm_xargs`, so the engine
  draws the canvas itself — is a *control*, and it prints nats on every boot that
  exists. It must therefore be unable to fail a run (#2116 clause 4): the shipped
  probe exited 1 on the `BATCH_INVARIANT=1` boot whose seeded reads measured
  0.0000, which is why clause 4 of #1357 could never be reported.
* A host with no djev on it must **pass** (`exit 0`) with an explicit
  `engine unreachable` line. GPU 2 is single-tenant (`start-djev.sh` header):
  the Qwen3.6 secondary owns it on some boots, and `:8011/health` is the signal
  production itself uses for that (`app/supervisor_client.py`,
  `scripts/service_health_check.py`). A probe that returned failure there would
  report a healthy machine as broken, which is the same class of defect as the
  one #1357 is bisecting.

The scripted engine also lets the tests pin the thing a real one cannot: that the
replayed requests are actually byte-identical, that every canvas position of a
production-length read is scored, that the seeded shape is the shape production
sends — the *same bytes* `eval/djev/seeded_canvas_probe.py` sends — and that the
prefix-cache counters the probe prints are read from the engine rather than
assumed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.djev_canvas_shape import (
    CANVAS_READ_POSITIONS, seeded_read_xargs, seed_canvas)
from scripts.djev_determinism_probe import (
    HEADER_ROW_PLACEHOLDER, HITS_TOTAL, QUERIES_TOTAL, completions_body, default_prompt,
    main, max_abs_delta, max_abs_delta_read, parse_counters, read_positions)
# The row the probe now emits is graded by the same expression that grades the
# script header it is pasted into, so the two cannot drift apart.
from tests.test_start_djev_flags import VARIANT_RE

ROOT = Path(__file__).resolve().parents[1]
EVAL_PROBE = ROOT / "eval" / "djev" / "seeded_canvas_probe.py"


class FakeEngine:
    """Answers /health, /metrics and /v1/completions from a scripted read.

    `script` is either one script for both read shapes, or
    `{"seeded": [...], "unseeded": [...]}` to give each shape its own — which is
    the whole point of #2116: the real engine answers those two differently, and
    a fake that ignored `vllm_xargs` could not express the case the exit code now
    keys off. Each entry is one read: a top_logprobs dict for one position, or a
    list of them for a production-length read of `CANVAS_READ_POSITIONS`. Entries
    cycle once exhausted.

    Counters move with every completion the way a real engine's do: `queries` by
    the prompt length, and `hits` only by the part of the prompt the engine had
    ALREADY seen — the request that first carries a prompt is the one that
    populates the cache, so it reports a miss (live engine, warm regime: run 1
    `hits+0`, run 2 `hits+4768` of 4800). A fake that credited every unsalted
    request would let the tests assert a warm baseline that no engine produces,
    which is the confound this instrument exists to remove.
    """

    # A prefix hit is whole reusable blocks only, and the block holding the
    # position being generated cannot be reused: `((tokens - 1) // 32) * 32`.
    # That reproduces both numbers the live engine printed — 4800 tokens ->
    # hits+4768 and 180 tokens -> hits+160 — which is why CHUNK is 32 and not a
    # guess: the fixture has to make the boundary the engine makes, or a test can
    # assert a cache delta no boot produces.
    CHUNK = 32

    def __init__(self, script, *, no_logprobs=False):
        if isinstance(script, dict):
            self.scripts = {k: list(v) for k, v in script.items()}
        else:
            self.scripts = {"seeded": list(script), "unseeded": list(script)}
        self.no_logprobs = no_logprobs
        self.calls = 0
        self.calls_by_shape = {"seeded": 0, "unseeded": 0}
        self.bodies: list[dict] = []
        self.queries = 0.0
        self.hits = 0.0
        self.seen_prompts: set[tuple[str, str, str]] = set()
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
                # The engine answers the seeded read out of a fixed canvas and the
                # unseeded one out of torch.randint, so it is not the same answer.
                shape = "seeded" if body.get("vllm_xargs") else "unseeded"
                salt = body.get("cache_salt")
                # Keyed per read shape as well as prompt+salt: a seeded read's
                # canvas tokens are part of the input its blocks are hashed from,
                # so its prefixes are not the ones an unseeded read of the same
                # prompt populates. Each shape therefore has its own first-request
                # miss, the same one the live engine shows within a shape.
                key = (shape, body.get("prompt", ""), "" if salt is None else str(salt))
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
                script = outer.scripts[shape]
                read = script[outer.calls_by_shape[shape] % len(script)]
                outer.calls_by_shape[shape] += 1
                outer.calls += 1
                positions = [read] if isinstance(read, dict) else list(read)
                choice: dict = {"index": 0, "text": "\n", "finish_reason": "length"}
                if not outer.no_logprobs:
                    choice["logprobs"] = {"tokens": ["\n"] * len(positions),
                                          "top_logprobs": positions}
                self._send(200, {
                    "id": "cmpl-x", "model": "djev", "object": "text_completion",
                    "choices": [choice],
                    "usage": {"prompt_tokens": tokens,
                              "completion_tokens": len(positions),
                              "total_tokens": tokens + len(positions)},
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

#: One position's top-k, and the same one shifted by exactly `SHIFT` nats. Every
#: value here is exactly representable in binary (halves and quarters), because
#: these tests assert a delta to four decimals *and* as a float, and a fixture
#: built from -0.1 would make the assertion depend on rounding rather than on the
#: probe. Same token set in both, so a read built from them differs in values only
#: and the top-k order is provably unchanged — which is what makes the delta
#: attributable to the position it was put in rather than to a reordering.
POS = {" the": -0.5, " a": -7.0, " no": -8.25}
SHIFT = 0.25


def read_of(*, shift_at: int | None = None) -> list[dict]:
    """A production-length read (16 positions); position `shift_at` moved 0.5 nats.

    The default has every position identical, which a real read never is — but the
    tests below need a fixture whose ONLY difference is the position they name.
    """
    pos = [dict(POS) for _ in range(CANVAS_READ_POSITIONS)]
    if shift_at is not None:
        pos[shift_at] = {t: v - SHIFT for t, v in POS.items()}
    return pos


def parsed(read: list[dict]) -> list[list[tuple[str, float]]]:
    """A scripted read in the shape `read_positions` hands the fold: one list of
    `(token, logprob)` pairs per canvas position, in the order the engine sent."""
    return [list(pos.items()) for pos in read]


def _request_lines(out: str, shape: str, regime: str) -> list[str]:
    """The probe's per-request lines for one shape+regime."""
    return [ln for ln in out.splitlines()
            if re.match(rf"^{shape}\s+{regime}\s+run \d", ln)]


def _pair(out: str, shape: str) -> re.Match | None:
    return re.search(rf"^{shape} warm/cold max \|delta\|(?: \(control\))? = (.+) nats",
                     out, re.MULTILINE)


def _measured(out: str, shape: str, regime: str) -> str:
    m = re.search(rf"^{shape} {regime} max \|delta label logprob\| = (\d+\.\d+) nats",
                  out, re.MULTILINE)
    assert m, f"no {shape} {regime} summary line in:\n{out}"
    return m.group(1)


# ── Clause 1: the outgoing body is the production shape, from one builder ─────

def test_the_seeded_request_carries_the_production_vllm_xargs():
    """The read that the exit code keys off must be the read production makes: a
    seeded canvas, an explicit canvas length, one step, read-only (#2116 clause 1).
    Sent bare, the engine draws the canvas itself and the probe is measuring that
    draw — which is why it exited 1 on the boot whose seeded reads were 0.0000."""
    with FakeEngine(STABLE) as eng:
        _run(eng.url, eng.url, "--runs", "2", "--regime", "warm")
    seeded = [b for b in eng.bodies if "vllm_xargs" in b]
    unseeded = [b for b in eng.bodies if "vllm_xargs" not in b]
    assert len(seeded) == 2 and len(unseeded) == 2, \
        f"both shapes must be sent, one seeded one not: {[sorted(b) for b in eng.bodies]}"
    for b in seeded:
        x = b["vllm_xargs"]
        assert x["diffusion_max_steps"] == 1, x
        assert x["diffusion_read_only"] is True, x
        assert x["diffusion_canvas_length"] == CANVAS_READ_POSITIONS, x
        assert len(x["diffusion_seed_canvas"]) == CANVAS_READ_POSITIONS, x
        assert b["max_tokens"] == CANVAS_READ_POSITIONS, \
            "a seeded read scored one position is not production's read either"


def test_the_canvas_shape_builder_is_shared_with_the_eval_instrument(monkeypatch,
                                                                    capsys):
    """Two copies of four engine arguments is how #2116 happened: the graded probe
    and `eval/djev/seeded_canvas_probe.py` each held their own, and only one of
    them was ever the production shape. Both now call
    `scripts.djev_canvas_shape`, and importing that eval module must not fire
    requests — it used to do its whole job at import time, which is why it could
    not simply be imported here."""
    def refuse(*a, **k):
        raise AssertionError("importing seeded_canvas_probe.py issued a request")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    spec = importlib.util.spec_from_file_location("scp_under_test", EVAL_PROBE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # must not raise, must not print
    assert capsys.readouterr().out == "", "importing the eval probe printed"

    # Same builder objects, not two functions that happen to agree today.
    assert mod.seed_canvas is seed_canvas
    assert mod.seeded_read_xargs is seeded_read_xargs

    # And the bytes it puts on the wire are the bytes the graded probe puts on it.
    captured: list[dict] = []

    def record(req, *a, **k):
        captured.append(json.loads(req.data))

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *e): return False
            def read(self): return b'{"choices":[{"logprobs":{"top_logprobs":[{}]}}]}'
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", record)
    mod.call(True, None)
    probe_body = completions_body("seeded", model="djev", prompt=mod.PROMPT, logprobs=5)
    assert captured, "the eval probe's call() did not issue a request"
    assert captured[0]["vllm_xargs"] == probe_body["vllm_xargs"], \
        "the two instruments are sending different seeded reads"
    assert captured[0]["max_tokens"] == probe_body["max_tokens"]


# ── Clause 2: every canvas position is scored, not only the first ────────────

def test_a_disagreement_only_at_position_seven_fails_the_run():
    """Production reads 16 positions off one canvas, so a boot that repeats
    position 0 and moves at position 7 has answered a production question
    differently (#2116 clause 2). Scoring `top_logprobs[0]` alone — what the
    shipped probe did — prints 0.0000 and exits 0 for that boot."""
    with FakeEngine({"seeded": [read_of(), read_of(shift_at=7)],
                     "unseeded": [read_of(), read_of(shift_at=7)]}) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3", "--regime", "warm")
    assert rc == 1, f"a read that moves at position 7 is not deterministic: rc={rc}\n{out}"
    assert f"seeded warm max |delta label logprob| = {SHIFT:.4f} nats" in out, out
    assert "set changed: no" in out, "the fixture moves values, not order"
    # The printed sample is the position the number came from, so a human reading
    # the line can see the labels that produced it rather than position 0's.
    moving = _request_lines(out, "seeded", "warm")[1]
    assert f"' the':{POS[' the'] - SHIFT:+.4f}" in moving, \
        f"run 2 should show the moving position: {moving}"


def test_the_probe_returns_and_folds_every_canvas_position_it_received():
    """Engine side of clause 2: the probe hands back all 16 positions rather than
    `top_logprobs[0]`, which is what the shipped version did."""
    # One scripted read: a list of 16 positions, so the entry is a full read.
    with FakeEngine([read_of(shift_at=3)]) as eng:
        positions, prompt_tokens = read_positions(eng.url, body=completions_body(
            "seeded", model="djev", prompt="x", logprobs=5))
    assert prompt_tokens == 1
    assert len(positions) == CANVAS_READ_POSITIONS, \
        f"a production read is {CANVAS_READ_POSITIONS} positions, got {len(positions)}"
    assert positions[3] == list(read_of(shift_at=3)[3].items()), positions[3]
    assert positions[0] == list(POS.items()), positions[0]


def test_the_read_fold_is_over_every_position_and_a_short_read_is_a_change():
    """Logic side of clause 2: the fold walks the whole read, with the position
    index named so a caller cannot quietly compare only the head of it."""
    same = parsed(read_of())
    assert max_abs_delta_read(same, parsed(read_of())) == (0.0, False)
    assert max_abs_delta_read(same, parsed(read_of(shift_at=15))) == (SHIFT, False)
    # A read that came back shorter is a different answer, not a shorter diff.
    assert max_abs_delta_read(same, same[:1])[1], \
        "a read whose position count changed must not read as a smaller comparison"


# ── Clause 3: one run prints both pairs, each request carries its cache delta ─

def test_one_run_prints_the_seeded_gate_and_the_unseeded_control_pair():
    """The bisect needs both numbers in one report and no ambiguity about which is
    the verdict: the seeded pair is the gate, the unseeded pair is the canvas-RNG
    spread that sits beside it as a control (#2116 clause 3)."""
    with FakeEngine({"seeded": STABLE, "unseeded": DRIFTING}) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3")
    assert rc == 0, f"a moving control must not fail a passing gate: rc={rc}\n{out}"
    gate, control = _pair(out, "seeded"), _pair(out, "unseeded")
    assert gate and control, f"both labelled pairs expected:\n{out}"
    assert gate.group(1) == "0.0000 / 0.0000", gate.group(0)
    # 1.2900 is exactly the #1357 signature the DRIFTING fixture carries.
    assert control.group(1) == "1.2900 / 1.2900", control.group(0)
    assert "(control)" in control.group(0)
    # Per-request evidence survives the split: 2 shapes x 2 regimes x 3 runs.
    per_request = [ln for ln in out.splitlines() if " run " in ln]
    assert len(per_request) == 2 * 2 * 3, out
    for ln in per_request:
        assert "queries+" in ln and "hits+" in ln, \
            f"every request line carries its own cache delta: {ln}"


# ── Clause 4: the exit code is the seeded pair, and only that ────────────────

def test_a_nonzero_unseeded_control_cannot_fail_a_seeded_pass():
    """The whole reason #2116 exists. On the shipped `BATCH_INVARIANT=1` boot the
    seeded reads repeat at 0.0000 while the unseeded reads move by ~4.9 nats, and
    the graded probe reported that as a failure, so #1357's clause 4 could never
    be reported by any round or nightly."""
    with FakeEngine({"seeded": STABLE, "unseeded": DRIFTING}) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3")
    assert rc == 0, f"the control moved but the gate passed; rc must be 0: rc={rc}\n{out}"
    assert "DETERMINISTIC" in out
    assert "NOT DETERMINISTIC" not in out
    assert "1.2900" in out, f"the control's number is still printed: {out}"


def test_a_seeded_disagreement_fails_even_with_a_stable_control():
    """The other half of the keying: the gate must not be able to hide behind a
    quiet control."""
    with FakeEngine({"seeded": DRIFTING, "unseeded": STABLE}) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3")
    assert rc == 1, f"a seeded disagreement must fail: rc={rc}\n{out}"
    assert "the seeded production-shaped read disagrees by 1.2900 nats" in out, out
    assert "NOT DETERMINISTIC" in out


def test_the_control_alone_decides_nothing_and_earns_no_row():
    """`--shape unseeded` is a diagnostic. If it could print a header row, a
    canvas-RNG number lands in the verdict column of `start-djev.sh`'s table."""
    with FakeEngine({"seeded": STABLE, "unseeded": DRIFTING}) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2", "--shape", "unseeded")
    assert rc == 0, f"the control cannot fail a run: rc={rc}\n{out}"
    assert not _header_rows(out), f"a control-only run has no gate to record:\n{out}"
    assert "no gate measured" in out, out


# ── The rest of #1357's contract, unchanged ──────────────────────────────────

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
    for shape in ("seeded", "unseeded"):
        warm = _request_lines(out, shape, "warm")
        assert len(warm) == 3, out
        assert "queries+2176 hits+0" in warm[0], \
            f"the first warm request populates the cache and gets no hits: {warm[0]}"
        for line in warm[1:]:
            assert "queries+2176 hits+2144" in line, \
                f"a repeated warm request reuses the aligned block: {line}"
    with FakeEngine(STABLE) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "2", "--regime", "cold")
    assert rc == 0, out
    assert out.count("queries+2176 hits+0") == 2 * 2, \
        "a fresh salt per request must never report a hit, or the cold regime is warm"


def test_every_run_sends_the_same_bytes_apart_from_the_cold_salt():
    """If the prompt drifted, the probe would be measuring prompt length and
    calling it kernels — which is the trap #1357's triage named for whoever
    picks this up."""
    with FakeEngine(STABLE) as eng:
        _run(eng.url, eng.url, "--runs", "2", "--regime", "cold",
             "--shape", "seeded")
    assert len(eng.bodies) == 2
    prompts = {b["prompt"] for b in eng.bodies}
    assert len(prompts) == 1, "the probe varied its own input between runs"
    # max_tokens is production's read length, not 1: #2116 clause 2 replaced the
    # shipped single-position read, so this is the assertion that used to pin it.
    assert all(b["max_tokens"] == CANVAS_READ_POSITIONS and b["logprobs"] == 5
               for b in eng.bodies)
    assert len({b["cache_salt"] for b in eng.bodies}) == 2, \
        "the cold regime must salt every request, or the prefix cache is being measured instead"
    canvases = {tuple(b["vllm_xargs"]["diffusion_seed_canvas"]) for b in eng.bodies}
    assert len(canvases) == 1, \
        "a seeded read that redraws its canvas per request is the unseeded read again"


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
        f"the built-in default prompt is {tokens} tokens, inside the "
        f"2048 compile_ranges_endpoints the production prompt is past")
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
    assert "the seeded read's top-k itself changed run to run" in out


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


# ── #1363 clause 2: a row the sweep can paste without hand-formatting ────────

def _header_rows(out: str) -> list[str]:
    """The lines shaped like a `start-djev.sh` header row, placeholder and all."""
    return [ln for ln in out.splitlines() if ln.startswith("#   ")]


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
    assert m["warm"] == _measured(out, "seeded", "warm"), \
        f"row warm must be the measured seeded warm: {row}"
    assert m["cold"] == _measured(out, "seeded", "cold"), \
        f"row cold must be the measured seeded cold: {row}"
    assert m["inv"] == "0", f"BATCH_INVARIANT defaults to 0, the shipped value: {row}"


def test_the_row_is_the_seeded_gate_and_never_the_control():
    """The table's verdict column. A boot whose seeded pair is 0.0000 while its
    control moves must reach the header as the pair it earned; taking the control
    instead would record a passing boot as failing — the same error #2116 removes
    from the exit code, one surface up."""
    with FakeEngine({"seeded": STABLE, "unseeded": DRIFTING}) as eng:
        rc, out = _run(eng.url, eng.url, "--runs", "3", "--variant-label", "inv1-shipped")
    assert rc == 0, out
    row = _header_rows(out)[0]
    m = VARIANT_RE.match(row.replace(HEADER_ROW_PLACEHOLDER, "500", 1))
    assert m, row
    assert m["cold"] == "0.0000" and m["warm"] == "0.0000", \
        f"the control's 1.2900 must not be pasted into the verdict column: {row}"


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


# ── No engine is not a failure ───────────────────────────────────────────────

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
    report 0 queries and 0 hits and look exactly like a constant cache. Two
    series per counter, because that is what a real exposition carries and the
    summing is the thing that must not silently under-count."""
    text = ('# TYPE vllm:prefix_cache_queries_total counter\n'
            'vllm:prefix_cache_queries_total{model_name="djev",engine_index="0"} 4352.0\n'
            'vllm:prefix_cache_queries_total{model_name="djev",engine_index="1"} 444.0\n'
            'vllm:prefix_cache_hits_total{model_name="djev",engine_index="0"} 4288.0\n'
            'vllm:prefix_cache_hits_total{model_name="djev",engine_index="1"} 11.0\n')
    got = parse_counters(text)
    assert got[QUERIES_TOTAL] == 4796.0
    assert got[HITS_TOTAL] == 4299.0
