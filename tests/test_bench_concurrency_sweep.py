"""The concurrency sweep client, tested against a fake engine that serves what we say.

Every clause of #2565 here is about the INSTRUMENT, not the engine: a client that
sweeps an explicit list of levels, resolves its model id from the server, decides
contamination from the engine's own counters, and refuses to write an artifact whose
conditions it cannot pin. So the fixture is a vLLM-shaped HTTP server on an
ephemeral 127.0.0.1 port (the `tests/test_djev_determinism_probe.py` fake-engine
idiom) whose `/metrics` values are numbers this file sets. The engine's real knee is
a drained-window measurement that a round cannot take — four worker turns were in
flight during triage — and no fixture substitutes for it: what these nodes prove is
that when the real run happens, the table it writes is the table the item asked for.

The engine FLAGS are fake the same way but through `/proc`, not the network:
`--proc-root` points at a fixture directory holding one numeric pid whose `cmdline`
names the fixture's own port. That is the seam the artifact's header is built from,
so the pinned `max_num_seqs=8` is read off argv exactly as production does it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "agent-services" / "bin" / "bench-35b-concurrency.py"

_spec = importlib.util.spec_from_file_location("bench_35b_concurrency", SCRIPT)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)

SOURCE = SCRIPT.read_text(encoding="utf-8")

#: The model the module used to name at module scope. It is not served: `:8096`
#: answers `Qwen3.8-Flash-Next-nvfp4` and the alias `primary`.
RETIRED_MODEL = "Qwen3.6-35B-A3B-nvfp4"
SERVED = ["Qwen3.8-Flash-Next-nvfp4", "primary"]

#: A serving argv of the shape `start-qwen38-flash-next.sh` actually builds, with the
#: two flags the sweep's header is written to pin: `--max-num-seqs` (the script's
#: `MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"` at :356, passed at :640) and `--speculative-config`.
#: Distinct pids per fixture, so one node can stand up two engines over one tmp_path.
_pid_cycle = iter(range(4242, 4242 + 10_000, 2))

#: Distinct --out directories per `_run_main` call inside one node.
_run_cycle = iter(range(1, 10_000))

SERVING_ARGS = [
    "/home/alansrobotlab/lloyd/.venvs/vllm-primary/bin/python", "-m", "vllm.serve",
    "--model", "/models/Inferact-Qwen3.8-Flash-Next-NVFP4",
    "--served-model-name", "primary",
    "--port", None,                              # filled with the fixture's port
    "--host", "127.0.0.1",
    "--max-model-len", "262144",
    "--max-num-seqs", "8",
    "--gpu-memory-utilization", "0.82",
    "--enable-prefix-caching",
    "--scheduling-policy", "priority",
    "--async-scheduling",
    "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}',
    "--kv-cache-dtype", "fp8",
    "--max-num-batched-tokens", "16384",
]


class FakeEngine:
    """A vLLM endpoint with hand-set counters, plus the pool's status route.

    `metrics` is mutable and read per request, so a node can move a counter between a
    level's before-snapshot and its after-snapshot, or hold `num_requests_running`
    above the level the sweep launched to stand in for traffic from somewhere else.

    Two knobs the #2786 clauses drive:

    * `prefix={"queries": q, "hits": h, "d_queries": dq, "d_hits": dh}` serves the two
      `vllm:prefix_cache_*` counters at the given lifetimes and moves them by the given
      amount on every completed chat request. It is OPT-IN: the default fixture is an
      engine that publishes NEITHER counter, which is the state a level row has to
      report as `null` rather than as zero reuse. `d_queries: 0` is the other case that
      matters — a level with no queries to divide by must publish the pair and no ratio.
    * `tokenize_count=N` answers `/tokenize` with `{"count": N}` (the route
      `bench-admission-stall.py:107` posts to) and records the bodies in
      `tokenize_calls`; `None` is an engine with no such route, which is a refusal.
    """

    def __init__(self, ids=SERVED, metrics=None, status=None, tokens=8,
                 token_delay=0.002, preemption_step=1.0, prefix=None,
                 tokenize_count=4212):
        self.ids = list(ids)
        self.metrics = dict({
            "vllm:kv_cache_usage_perc": 0.42,
            "vllm:num_requests_running": 0.0,
            "vllm:num_requests_waiting": 0.0,
            "vllm:num_preemptions_total": 7.0,
            bc.SPEC_ACCEPTED: 9246470.0,
            bc.SPEC_DRAFTED: 15837146.0,
        }, **(metrics or {}))
        self.status = dict({"initialized": True, "pool": {
            "running": True, "paused": False, "paused_by": None,
            "in_flight_count": 0, "slots": 6}}, **(status or {}))
        self.tokens = tokens
        self.token_delay = token_delay
        self.preemption_step = preemption_step
        self.prefix = prefix
        if prefix:
            self.metrics.setdefault(bc.PREFIX_QUERIES, float(prefix["queries"]))
            self.metrics.setdefault(bc.PREFIX_HITS, float(prefix["hits"]))
        self.tokenize_count = tokenize_count
        self.tokenize_calls: list[dict] = []
        self.chat_requests: list[dict] = []
        self.lock = threading.Lock()

    # ── the wire ────────────────────────────────────────────────────────────
    def _sse(self, send):
        for i in range(self.tokens):
            payload = {"choices": [{"delta": {"content": f"tok{i} "}}]}
            send(f"data: {json.dumps(payload)}\n\n")
        send(f"data: {json.dumps({'usage': {'completion_tokens': self.tokens}, 'choices': []})}\n\n")
        send("data: [DONE]\n\n")

    def handler(self):
        eng = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):          # silence
                pass

            def _send(self, code, body, ctype="application/json"):
                raw = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, json.dumps(
                        {"object": "list",
                         "data": [{"id": i, "object": "model"} for i in eng.ids]}))
                elif self.path == "/metrics":
                    with eng.lock:
                        lines = [f"{name}{{label=\"sweep\"}} {value}"
                                 for name, value in eng.metrics.items()]
                        # A series the sweep must NOT read as one of its four:
                        # label-summing has to pick the named counter, not the file.
                        lines.append('vllm:something_else{label="x"} 999')
                    self._send(200, "\n".join(lines) + "\n", "text/plain")
                elif self.path == "/api/workers/status":
                    self._send(200, json.dumps(eng.status))
                else:
                    self._send(404, json.dumps({"error": self.path}))

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/tokenize":
                    with eng.lock:
                        eng.tokenize_calls.append(body)
                    if eng.tokenize_count is None:
                        self._send(404, json.dumps({"error": self.path}))
                    else:
                        self._send(200, json.dumps({"count": eng.tokenize_count,
                                                    "tokens": []}))
                    return
                if self.path != "/v1/chat/completions":
                    self._send(404, json.dumps({"error": self.path}))
                    return
                with eng.lock:
                    eng.chat_requests.append({
                        "model": body.get("model"),
                        "max_tokens": body.get("max_tokens"),
                        "prompt": (body.get("messages") or [{}])[0].get("content", ""),
                        "priority": body.get("priority"),
                    })
                    eng.metrics["vllm:num_preemptions_total"] = (
                        eng.metrics["vllm:num_preemptions_total"] + eng.preemption_step)
                    if eng.prefix:
                        eng.metrics[bc.PREFIX_QUERIES] += eng.prefix["d_queries"]
                        eng.metrics[bc.PREFIX_HITS] += eng.prefix["d_hits"]

                def send(chunk):
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
                    if eng.token_delay:
                        import time
                        time.sleep(eng.token_delay)

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    eng._sse(send)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler


@contextmanager
def fake_engine(tmp_path, make_proc_entry=True, serving_args=SERVING_ARGS, **kw):
    eng = FakeEngine(**kw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), eng.handler())
    port = server.server_address[1]
    eng.base_url = f"http://127.0.0.1:{port}"
    eng.status_url = f"{eng.base_url}/api/workers/status"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    proc = tmp_path / "proc"
    if make_proc_entry:
        argv = [str(port) if tok is None else tok for tok in serving_args]
        eng.pid = 4242 + next(_pid_cycle)
        (proc / str(eng.pid)).mkdir(parents=True, exist_ok=True)
        (proc / str(eng.pid) / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    else:
        eng.pid = None
    eng.proc_root = proc
    thread.start()
    try:
        yield eng
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ── clause 1: an explicit list of levels, per repeat, with a spread ─────────────

def test_the_sweep_runs_every_level_of_a_requested_list_not_the_old_fixed_pair(tmp_path):
    """Clause 1: the batch phase is gone, and so is the list it could not express.

    `bench-35b-concurrency.py` had `single()` at concurrency 1 and `batch()` at 4 and
    8, and printed the mean of the batch's two runs — a curve with three points and no
    idea of its own noise. This asks for 1, 3 and 5 — no 4, no 8 — three times each,
    and requires the table to come back with all three levels and three runs apiece.
    """
    with fake_engine(tmp_path) as eng:
        doc = bc.sweep(eng.base_url, [1, 3, 5], 3, 16, "primary", "primary",
                       bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                       "drained", monitor_interval=0.05, warmup=False)
    levels = [lvl["level"] for lvl in doc["levels"]]
    assert levels == [1, 3, 5], levels
    assert doc["requested_levels"] == [1, 3, 5], doc["requested_levels"]
    for lvl in doc["levels"]:
        assert lvl["repeats"] == 3, lvl
        assert len(lvl["runs"]) == 3, lvl
        assert lvl["n_err"] == 0, lvl

    # Every level reports the three numbers the knee is read from, per repeat.
    for lvl in doc["levels"]:
        for run in lvl["runs"]:
            assert run["aggregate_generation_tokens_per_s"] > 0, run
            assert run["ttft_mean_ms"] > 0, run
            assert run["ttft_p95_ms"] >= run["ttft_mean_ms"], run


def test_each_level_publishes_the_max_minus_min_spread_across_its_repeats(tmp_path):
    """Clause 1's noise term: a mean without a spread is not a result.

    The stage that this method comes from stated a ±2 tok/s run-to-run floor, and the
    item's acceptance says a movement inside the repeat spread is reported as no
    effect — so the spread has to be in the artifact and be the real max-minus-min of
    the repeats, recomputed here from the runs beside it rather than trusted.
    """
    with fake_engine(tmp_path) as eng:
        doc = bc.sweep(eng.base_url, [2], 3, 16, "primary", "primary",
                       bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                       "drained", monitor_interval=0.05, warmup=False)
    lvl = doc["levels"][0]
    for key in ("aggregate_generation_tokens_per_s", "ttft_mean_ms", "ttft_p95_ms"):
        vals = [run[key] for run in lvl["runs"] if run.get(key) is not None]
        assert vals, lvl
        assert lvl[f"{key}_spread"] == round(max(vals) - min(vals), 1), (key, lvl)
        assert lvl[f"{key}_spread"] >= 0, (key, lvl)
    assert lvl["aggregate_generation_tokens_per_s_spread"] >= 0


def test_the_level_list_is_parsed_as_a_list_and_refuses_an_ambiguous_one():
    """The flag has to take a list, and refuse the shapes that would silently sweep
    the wrong thing: a non-integer, a repeated level, or nothing."""
    assert bc.parse_levels("1,2,4,6,8,12,16") == [1, 2, 4, 6, 8, 12, 16]
    assert bc.parse_levels(" 1 , 3 ") == [1, 3]
    for bad in ("1,1,2", "0", "1,,2", "4,x", ""):
        with pytest.raises(bc.SweepError):
            bc.parse_levels(bad)


# ── clause 2: the model id comes from the server ───────────────────────────────

def test_the_model_id_comes_from_the_served_list_and_a_wrong_one_is_a_refusal(tmp_path):
    """Clause 2: no module-level model constant naming a retired engine.

    The retired id was `RETIRED_MODEL`; `:8096/v1/models` answers `SERVED`. The
    default has to be the server's own first id, `--model` has to be checked against
    that list, and the name has to be gone from this file outside the docstring that
    explains why it went.
    """
    assert not hasattr(bc, "MODEL"), "a module-level model constant is back"
    body = SOURCE.replace(ast.get_docstring(ast.parse(SOURCE)) or "", "")
    assert RETIRED_MODEL not in body, (
        "the retired id is in the code, not just in the prose that names it")

    with fake_engine(tmp_path) as eng:
        resolved = bc.resolve_model(eng.base_url)
        assert resolved["model"] == SERVED[0], resolved
        assert resolved["served_ids"] == SERVED, resolved
        assert bc.resolve_model(eng.base_url, "primary")["model"] == "primary"
        with pytest.raises(bc.SweepError) as err:
            bc.resolve_model(eng.base_url, RETIRED_MODEL)
        assert "not what" in str(err.value), err.value
        assert SERVED[0] in str(err.value), (
            "the refusal must name what IS served, so the operator can pick it")


def test_the_sweep_asks_the_resolved_model_on_every_request(tmp_path):
    """The resolved id is what goes on the wire, not a name from the source."""
    with fake_engine(tmp_path) as eng:
        bc.sweep(eng.base_url, [1], 1, 16, "primary", "primary",
                 bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                 "drained", monitor_interval=0.05, warmup=False)
        assert eng.chat_requests, "no request reached the engine at all"
        assert {r["model"] for r in eng.chat_requests} == {"primary"}
        assert all(r["priority"] == 1 for r in eng.chat_requests), (
            "a sweep at the default priority preempts live work and forces its "
            "prefills to recompute — the contamination would be self-inflicted")


# ── clause 3: engine counters, and a contaminated level is not in the table ────

def test_each_level_snapshots_the_four_engine_counters_before_and_after(tmp_path):
    """Clause 3, first half: the engine's own reading of the level, around it.

    The preemption delta is the difference between "the engine batched it" and "the
    engine thrashed and recompute-paid for it", so it has to be a difference of the
    engine's counter across the level, not an absolute. The fixture moves
    `vllm:num_preemptions_total` by one per completed request, so the delta is
    countable from the request log beside it.
    """
    with fake_engine(tmp_path) as eng:
        doc = bc.sweep(eng.base_url, [2], 1, 16, "primary", "primary",
                       bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                       "drained", monitor_interval=0.05, warmup=False)
    lvl = doc["levels"][0]
    for key in ("kv_cache_usage", "num_requests_running", "num_requests_waiting",
                "preemptions_total"):
        assert key in lvl["engine_before"], lvl["engine_before"]
        assert key in lvl["engine_after"], lvl["engine_after"]
    assert lvl["engine_before"]["kv_cache_usage"] == 0.42, lvl["engine_before"]
    assert lvl["engine_before"]["preemptions_total"] == 7.0, lvl["engine_before"]
    n_requests = len(eng.chat_requests)
    assert lvl["engine_after"]["preemptions_total"] == (
        7.0 + eng.preemption_step * n_requests), (lvl["engine_after"], n_requests)
    assert lvl["preemptions_during"] == eng.preemption_step * n_requests, lvl


def test_a_level_where_the_engine_held_more_than_the_sweep_launched_is_excluded(tmp_path):
    """Clause 3, second half: contamination is decided and the level is dropped.

    `num_requests_running` sitting at 9 while the sweep runs a level of 1 is another
    workload in the engine — the 2026-09-08 case, where a "single-stream" measurement
    of this slot read 50-100 tok/s against a recorded 182 and the number looked fine.
    Such a level must not sit in the table the knee is read from, and must be visible
    as excluded rather than silently absent.
    """
    with fake_engine(tmp_path, metrics={"vllm:num_requests_running": 9.0}) as eng:
        doc = bc.sweep(eng.base_url, [1, 2], 2, 16, "primary", "primary",
                       bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                       "drained", monitor_interval=0.02, warmup=False)
    assert doc["levels"] == [], doc["levels"]
    assert [lvl["level"] for lvl in doc["excluded_levels"]] == [1, 2]
    for lvl in doc["excluded_levels"]:
        assert lvl["contaminated"] is True, lvl
        assert lvl["observed_inflight_peak"] == 9.0, lvl

    # And the clean case: the same server with nothing else in flight keeps both.
    with fake_engine(tmp_path) as eng2:
        clean = bc.sweep(eng2.base_url, [1, 2], 2, 16, "primary", "primary",
                         bc.engine_config(eng2.base_url, eng2.proc_root),
                         eng2.status_url, "drained", monitor_interval=0.02,
                         warmup=False)
    assert [lvl["level"] for lvl in clean["levels"]] == [1, 2], clean["levels"]
    assert clean["excluded_levels"] == []
    assert all(lvl["contaminated"] is False for lvl in clean["levels"])


# ── clause 4: a dated artifact under the data root, and a refusal to write one ──

def _files(out: Path) -> list[Path]:
    """What a run wrote — and a refusal that creates no directory at all counts."""
    return sorted(out.iterdir()) if out.exists() else []


def _run_main(eng, tmp_path, *extra, tag: str = ""):
    """One out directory per call: a refusal is proved to write nothing only if the
    directory could not already hold another call's artifact."""
    out = tmp_path / f"sweep-out{tag or next(_run_cycle)}"
    return bc.main(["--base-url", eng.base_url, "--pool-status", eng.status_url,
                    "--proc-root", str(eng.proc_root), "--out", str(out),
                    "--levels", "1", "--repeats", "1", "--max-tokens", "16",
                    "--no-warmup", "--label", "pre", *extra]), out


def test_the_sweep_refuses_before_loading_an_engine_that_is_not_quiet(tmp_path):
    """Clause 4: the pool declaration is verified, and verified BEFORE the load.

    The sweep competes with live worker turns and voice turns for the resource it is
    measuring, so `--pool drained` is a claim the client checks against
    `/api/workers/status`, not a box the operator ticks. Three refusals, each writing
    nothing and issuing no request: jobs in flight against a `drained` declaration; a
    `paused` declaration the pool does not hold; and an unreadable pool.
    """
    busy = {"initialized": True, "pool": {"running": True, "paused": False,
                                          "in_flight_count": 3}}
    with fake_engine(tmp_path, status=busy) as eng:
        rc, out = _run_main(eng, tmp_path, tag="r1")
        assert rc == 2, rc
        assert _files(out) == [], "a refused run still wrote an artifact"
        assert eng.chat_requests == [], "the refusal happened AFTER loading the engine"

    with fake_engine(tmp_path) as eng:
        rc, out = _run_main(eng, tmp_path, "--pool", "paused", tag="r2")
        assert rc == 2, rc
        assert _files(out) == [], out
        assert eng.chat_requests == [], out

    # A pause the pool really holds satisfies the declaration — the check is on the
    # pool's state, not on the flag's presence.
    paused = {"initialized": True, "pool": {"running": True, "paused": True,
                                            "paused_by": "autonomy-#58",
                                            "in_flight_count": 0}}
    with fake_engine(tmp_path, status=paused) as eng:
        rc, out = _run_main(eng, tmp_path, "--pool", "paused", tag="r3")
        assert rc == 0, rc
        assert len(_files(out)) == 1, _files(out)

    with fake_engine(tmp_path, status={"initialized": False}) as eng:
        rc, out = _run_main(eng, tmp_path, tag="r4")
        assert rc == 2, rc
        assert _files(out) == [], out


def test_the_artifact_is_dated_under_the_data_root_and_pins_the_engine_config(tmp_path):
    """Clause 4: the header is the artifact's whole value, so every flag is in it.

    A curve is per-model, per-precision and per-config: the source measurement lost
    its own final comparison to a config swap noticed afterwards, which is why this
    header is read off the serving process's argv rather than typed in — a boot script
    says what a boot WOULD do, `/proc` says what this engine IS doing. Includes
    `max_num_seqs`, the one flag that turned 800 tok/s into 1,100 at concurrency 64.
    """
    with fake_engine(tmp_path) as eng:
        rc, out = _run_main(eng, tmp_path, tag="r5")
        assert rc == 0, rc
        files = _files(out)
        assert len(files) == 1, files
        path = files[0]
        assert re.fullmatch(r"concurrency-sweep-\d{8}T\d{6}Z-pre\.json", path.name), path.name
        doc = json.loads(path.read_text())

    assert doc["model"] == SERVED[0], doc["model"]
    assert doc["model_resolution"]["served_ids"] == SERVED
    assert doc["requested_levels"] == [1] and doc["repeats"] == 1
    cfg = doc["engine_config"]
    assert cfg["max_num_seqs"] == "8", cfg
    assert cfg["speculative_config"] == json.loads(
        '{"method": "mtp", "num_speculative_tokens": 3}'), cfg["speculative_config"]
    assert cfg["flags"]["max-num-batched-tokens"] == "16384", cfg["flags"]
    assert cfg["flags"]["enable-prefix-caching"] is True, cfg["flags"]
    assert cfg["flags"]["scheduling-policy"] == "priority", cfg["flags"]
    # Precision: no --quantization was passed, so the header says where it comes from
    # instead of writing a number nobody chose.
    assert "NVFP4" in str(cfg["quantization_or_precision"]), cfg["quantization_or_precision"]
    assert cfg["argv"][-1] == "--kv-cache-dtype" or "--kv-cache-dtype" in cfg["argv"], cfg["argv"]
    assert cfg["pid"] == eng.pid, cfg["pid"]
    assert "distinct prompts" in doc["prompt_policy"], doc["prompt_policy"]
    assert doc["pool_declaration"] == "drained", doc["pool_declaration"]
    assert doc["pool_at_write"]["in_flight_count"] == 0, doc["pool_at_write"]
    assert math.isclose(doc["spec_decode_counts"]["accepted_over_drafted"],
                        9246470 / 15837146, rel_tol=1e-9), doc["spec_decode_counts"]
    assert doc["written_at"].endswith("+00:00"), doc["written_at"]


def test_the_default_artifact_directory_is_the_data_root_eval_not_a_temporary(tmp_path,
                                                                              monkeypatch):
    """`/tmp` held the previous client's output, and `/tmp` is wiped.

    The eval baselines live under the data root, and so does anything a later round is
    meant to compare against — so the default goes through `app.paths.DATA_ROOT`,
    which is the resolution with rules, rather than a re-implemented `$LLOYD_DATA`.
    """
    from app import paths
    monkeypatch.delenv("LLOYD_DATA", raising=False)
    assert bc.default_out_dir() == Path(paths.DATA_ROOT) / "eval", bc.default_out_dir()
    assert str(bc.default_out_dir()).startswith(str(paths.DATA_ROOT)), bc.default_out_dir()


def test_a_config_this_file_cannot_pin_is_a_refusal_rather_than_a_blank(tmp_path):
    """An artifact with `max_num_seqs: unknown` is not comparable to anything, which
    is the failure mode the header exists to prevent. A serving process that does not
    pass the flag, a process table that cannot be read, and a port two processes
    claim all refuse the write; and the flag defaults to 8, so an artifact that says
    8 is distinguishable from one that says 128."""
    without_seqs = [tok for tok in SERVING_ARGS if tok not in ("--max-num-seqs", "8")]
    with fake_engine(tmp_path, serving_args=without_seqs) as eng:
        rc, out = _run_main(eng, tmp_path, tag="r6")
        assert rc == 2, rc
        assert _files(out) == [], out

    with fake_engine(tmp_path, make_proc_entry=False) as eng:
        rc, out = _run_main(eng, tmp_path, tag="r7")
        assert rc == 2, rc
        assert _files(out) == [], out

    with fake_engine(tmp_path) as eng:
        other = eng.proc_root / "9999"
        other.mkdir(parents=True)
        argv = [str(eng.base_url.rsplit(":", 1)[1]) if tok is None else tok
                for tok in SERVING_ARGS]
        (other / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
        rc, out = _run_main(eng, tmp_path, tag="r8")
        assert rc == 2, rc
        assert _files(out) == [], out


# ── the counters behind standing problem 7 ────────────────────────────────────

def test_the_spec_decode_pair_is_the_two_counters_the_doc_claims_are_running(tmp_path):
    """The same two counters `tests/test_lloyd_profile_doc_claims.py` recomputes,
    read the way the sweep reads them. Kept here because this file is the only place
    that reads `/metrics` for a purpose other than the dashboard, and a drift between
    the two readers is how problem 7 went stale in the first place."""
    with fake_engine(tmp_path) as eng:
        got = bc.spec_decode_counts(eng.base_url)
    assert got["accepted_tokens"] == 9246470.0, got
    assert got["drafted_tokens"] == 15837146.0, got
    assert math.isclose(got["accepted_over_drafted"], 9246470 / 15837146, rel_tol=1e-9)

    with fake_engine(tmp_path, metrics={bc.SPEC_DRAFTED: 0.0}) as eng:
        assert bc.spec_decode_counts(eng.base_url)["accepted_over_drafted"] is None, (
            "a zero denominator must not divide")


# ── #2786 clause 1: prefix-cache reuse is measured per level, or reported as null ──

def test_every_level_row_carries_the_prefix_cache_pair_delta_or_null(tmp_path):
    """Clause 1: the row carries the delta of `vllm:prefix_cache_queries_total` and
    `vllm:prefix_cache_hits_total` across that level, beside the four existing engine
    counters — and `null`, never `0`, on an engine that serves neither.

    Before this the file ASSERTED reuse was limited to a level's own repeats (docstring
    `:68-77`) while never reading the two counters that would show it, so the claim was
    unfalsifiable. The fixture moves each counter by a known amount per completed
    request, so the delta is recomputed from the request log beside the row rather than
    trusted. The unserved case is the load-bearing one: a level recorded as
    "0 queries" reads as zero reuse, and what it means is that the engine gave no
    evidence at all.
    """
    prefix = {"queries": 5000.0, "hits": 500.0, "d_queries": 130.0, "d_hits": 104.0}
    with fake_engine(tmp_path, prefix=prefix) as eng:
        doc = bc.sweep(eng.base_url, [2], 1, 16, "primary", "primary",
                       bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                       "drained", monitor_interval=0.05, warmup=False)
    lvl = doc["levels"][0]
    n_requests = len(eng.chat_requests)
    assert n_requests == 2, eng.chat_requests
    pc = lvl["prefix_cache_during"]
    assert pc["queries_delta"] == prefix["d_queries"] * n_requests, (pc, n_requests)
    assert pc["hits_delta"] == prefix["d_hits"] * n_requests, (pc, n_requests)
    assert lvl["engine_before"]["prefix_cache_queries_total"] == 5000.0, lvl
    assert lvl["engine_after"]["prefix_cache_queries_total"] == 5000.0 + 260.0, lvl
    assert lvl["engine_before"]["prefix_cache_hits_total"] == 500.0, lvl
    # Beside the four, in the same row — not in a separate report a reader has to join.
    for key in ("kv_cache_usage", "num_requests_running", "num_requests_waiting",
                "preemptions_total"):
        assert key in lvl["engine_after"], lvl["engine_after"]

    with fake_engine(tmp_path) as eng:                 # default: neither counter served
        doc = bc.sweep(eng.base_url, [1], 1, 16, "primary", "primary",
                       bc.engine_config(eng.base_url, eng.proc_root), eng.status_url,
                       "drained", monitor_interval=0.05, warmup=False)
    pc = doc["levels"][0]["prefix_cache_during"]
    assert pc["queries_delta"] is None, pc
    assert pc["hits_delta"] is None, pc
    assert "hit_rate" not in pc, "a ratio cannot be divided out of nothing"
    assert doc["levels"][0]["engine_before"]["prefix_cache_queries_total"] is None
    assert bc.prefix_delta({}, {}) == {"queries_delta": None,
                                       "hits_delta": None}, bc.prefix_delta({}, {})


# ── #2786 clause 2: one captured long prompt, one pass per level ─────────────────

#: Something recognisably not one of the eight built-in prompts, so a row that sent the
#: wrong thing cannot pass by accident.
LONG_PROMPT = "REPLAYED LLOYD TURN >>> " + "summarise the vault and file the finding. " * 40


def _prompt_file(tmp_path, text=LONG_PROMPT, name="captured-turn.txt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_long_prompt_file_run_sends_that_prompt_once_per_level(tmp_path):
    """Clause 2: `--long-prompt-file PATH` puts the file's bytes on every request and
    runs each requested level EXACTLY ONCE whatever `--repeats` says, labelling every
    row it produces `shape: long_prefill`; a run without the flag labels `shape: short`.

    `--repeats 3` with `--levels 1,3` is therefore 4 chat requests, not 12: n=1 is the
    whole reason the long rows are printed apart, and a warm-up pass would spend that
    single pass on a warm prefill (reuse engages after one cold pass —
    `bench-prefix-reuse.py` trap 2). The labels are what stop a long row being
    downloaded into the short curve by whoever reads the artifact next.
    """
    with fake_engine(tmp_path) as eng:
        pf = _prompt_file(tmp_path)
        rc, out = _run_main(eng, tmp_path, "--long-prompt-file", str(pf),
                            "--levels", "1,3", "--repeats", "3", tag="L1")
        assert rc == 0, rc
        doc = json.loads(_files(out)[0].read_text())
    assert doc["shape"] == "long_prefill", doc["shape"]
    assert doc["requested_levels"] == [1, 3], doc["requested_levels"]
    assert doc["repeats"] == 3 and doc["repeats_per_level"] == 1, doc["repeats"]
    assert [lvl["level"] for lvl in doc["levels"]] == [1, 3], doc["levels"]
    for lvl in doc["levels"]:
        assert lvl["shape"] == "long_prefill", lvl
        assert lvl["repeats"] == 1, lvl
    assert len(eng.chat_requests) == 4, [r["prompt"][:24] for r in eng.chat_requests]
    assert {r["prompt"] for r in eng.chat_requests} == {LONG_PROMPT}, (
        "the long shape must send the file verbatim on every slot of every level")

    with fake_engine(tmp_path) as eng2:
        rc2, out2 = _run_main(eng2, tmp_path, tag="L1b")
        assert rc2 == 0, rc2
        doc2 = json.loads(_files(out2)[0].read_text())
    assert doc2["shape"] == "short", doc2["shape"]
    assert all(lvl["shape"] == "short" for lvl in doc2["levels"]), doc2["levels"]


def test_a_long_shape_artifact_names_the_prompt_file_and_its_engine_token_count(tmp_path):
    """Clause 3: for a long-shape run `prompt_policy` names the prompt file path and
    the prompt's token count from the engine's own tokenize route — the one
    `bench-admission-stall.py:107` posts to (`/tokenize`: vLLM serves no
    `/v1/tokenize`, and the sibling file is the route verified on this box) — while a
    default run's `prompt_policy` stays the existing fixed string, character for
    character.

    `PROMPT_POLICY` is a constant, so it structurally cannot carry a path or a size;
    the size has to be the engine's because there is no tokenizer helper anywhere in
    this tree and chars/4 would be an invention recorded as a measurement. An engine
    with no tokenize route is therefore a refusal that writes nothing, not a blank.
    """
    with fake_engine(tmp_path) as eng:
        pf = _prompt_file(tmp_path, name="turn-42.md")
        rc, out = _run_main(eng, tmp_path, "--long-prompt-file", str(pf), tag="L2")
        assert rc == 0, rc
        doc = json.loads(_files(out)[0].read_text())
    assert str(pf) in doc["prompt_policy"], doc["prompt_policy"]
    assert str(eng.tokenize_count) in doc["prompt_policy"], doc["prompt_policy"]
    assert eng.tokenize_calls, "the token count was never asked of the engine"
    assert eng.tokenize_calls[0]["prompt"] == LONG_PROMPT, eng.tokenize_calls[0]
    assert eng.tokenize_calls[0]["model"] == SERVED[0], eng.tokenize_calls[0]

    with fake_engine(tmp_path) as eng2:
        rc2, out2 = _run_main(eng2, tmp_path, tag="L2b")
        assert rc2 == 0, rc2
        doc2 = json.loads(_files(out2)[0].read_text())
    assert doc2["prompt_policy"] == bc.PROMPT_POLICY, doc2["prompt_policy"]
    assert eng2.tokenize_calls == [], "a short run has no reason to call /tokenize"

    with fake_engine(tmp_path, tokenize_count=None) as eng3:
        pf3 = _prompt_file(tmp_path, name="turn-43.md")
        rc3, out3 = _run_main(eng3, tmp_path, "--long-prompt-file", str(pf3), tag="L2c")
        assert rc3 == 2, rc3
        assert _files(out3) == [], out3
        assert eng3.chat_requests == [], "the refusal must precede the load"

    with fake_engine(tmp_path) as eng4:
        rc4, out4 = _run_main(eng4, tmp_path, "--long-prompt-file",
                              str(tmp_path / "no-such-turn.txt"), tag="L2d")
        assert rc4 == 2, rc4
        assert _files(out4) == [], out4
        assert eng4.chat_requests == [], out4


# ── #2786 clause 4: n=1 rows carry no spread and never sit in the knee table ──────

#: The knee table's header, byte for byte as `main` has printed it since #2565. Written
#: out here rather than imported from the script: clause 4 requires the default short
#: table's columns to be UNCHANGED, so a edit that renames or reorders a column has to
#: fail this node, not quietly move the expected value along with it.
SHORT_TABLE_HEADER = (f"  {'lvl':>4}  {'agg tok/s':>10} {'spread':>7}  "
                      f"{'ttft mean ms':>13} {'spread':>7}  {'ttft p95 ms':>12} "
                      f"{'spread':>7}  {'preempt':>8} {'kv':>6}")


def test_a_long_prefill_row_carries_no_spread_and_prints_in_its_own_n_equals_one_table(tmp_path,
                                                                                       capsys):
    """Clause 4: a long-shape row carries NO numeric `{key}_spread` value and prints in
    a separate table that states `n=1`, while the default short table's columns and
    rows are unchanged.

    The hazard was live in code before this: `aggregate_level` wrote `{key}_spread` for
    ANY non-empty run list, so a single run yielded `0.0` and the print loop emitted the
    column unconditionally — a long row bolted onto the old path would have published
    "variance zero" for a level whose variance was never measured. Absence of the key is
    the fix, so the assertion is `not in`, and the long table's own header carries the
    n=1 that explains why nothing is there. The short table is pinned by its exact
    header line so "unchanged" is checkable rather than intended.
    """
    with fake_engine(tmp_path) as eng:
        pf = _prompt_file(tmp_path, name="turn-44.txt")
        rc, out = _run_main(eng, tmp_path, "--long-prompt-file", str(pf),
                            "--levels", "1,2", "--repeats", "3", tag="L3")
        assert rc == 0, rc
        doc = json.loads(_files(out)[0].read_text())
        long_out = capsys.readouterr().out
    for lvl in doc["levels"]:
        assert lvl["shape"] == "long_prefill", lvl
        for key in ("aggregate_generation_tokens_per_s", "ttft_mean_ms", "ttft_p95_ms",
                    "itl_mean_ms"):
            assert f"{key}_spread" not in lvl, (key, lvl)
            assert f"{key}_mean" in lvl, (key, lvl)
    assert "LONG_PREFILL SHAPE" in long_out, long_out
    assert "n=1 per level" in long_out, long_out
    assert f"{'agg tok/s':>10} {'spread':>7}" not in long_out, (
        "the knee table's spread columns must not print over a long row")
    assert SHORT_TABLE_HEADER not in long_out, long_out

    with fake_engine(tmp_path) as eng2:
        rc2, out2 = _run_main(eng2, tmp_path, tag="L3b")
        assert rc2 == 0, rc2
        short_out = capsys.readouterr().out
    assert SHORT_TABLE_HEADER in short_out, short_out
    assert "LONG_PREFILL SHAPE" not in short_out, short_out
    doc2 = json.loads(_files(out2)[0].read_text())
    assert "aggregate_generation_tokens_per_s_spread" in doc2["levels"][0], doc2["levels"][0]


# ── #2786 clause 5: a reuse ratio only where its condition can hold ───────────────

def test_a_reuse_ratio_appears_only_on_a_nonzero_queries_delta_and_under_its_condition(tmp_path,
                                                                                       capsys):
    """Clause 5: a ratio beside the counter pair appears only when the queries delta is
    non-zero, and only labelled with the condition `bench-prefix-reuse.py:11-25`
    records — hits/queries is a token-level reuse rate ONLY while
    `prefix_cache_queries_total == prompt_tokens_total`. Otherwise the raw pair is all
    that appears.

    The condition is not ceremony: on the 2026-09-06 MTP boot that note describes, the
    queries counter summed across KV-cache groups, one 60,005-token prompt logged
    673,179 queries (11.2x), and the ratio still read ~68% while real cross-request
    reuse was zero. So the fixture's zero-queries engine has to yield the pair with no
    ratio in the artifact AND no percentage or condition line in the printed table, and
    the measuring engine's ratio has to travel with the parity named beside it.
    """
    zero = {"queries": 5000.0, "hits": 500.0, "d_queries": 0.0, "d_hits": 0.0}
    with fake_engine(tmp_path, prefix=zero) as eng:
        pf = _prompt_file(tmp_path, name="turn-45.txt")
        rc, out = _run_main(eng, tmp_path, "--long-prompt-file", str(pf), tag="L4")
        assert rc == 0, rc
        doc = json.loads(_files(out)[0].read_text())
        zero_out = capsys.readouterr().out
    pc = doc["levels"][0]["prefix_cache_during"]
    assert pc["queries_delta"] == 0.0, pc
    assert pc["hits_delta"] == 0.0, pc
    assert "hit_rate" not in pc, pc
    assert "%*" not in zero_out, zero_out
    assert "prompt_tokens_total" not in zero_out, (
        "the parity condition must not be printed under a number that is not a rate")

    live = {"queries": 5000.0, "hits": 500.0, "d_queries": 130.0, "d_hits": 104.0}
    with fake_engine(tmp_path, prefix=live) as eng2:
        pf2 = _prompt_file(tmp_path, name="turn-46.txt")
        rc2, out2 = _run_main(eng2, tmp_path, "--long-prompt-file", str(pf2), tag="L5")
        assert rc2 == 0, rc2
        doc2 = json.loads(_files(out2)[0].read_text())
        live_out = capsys.readouterr().out
    pc2 = doc2["levels"][0]["prefix_cache_during"]
    assert math.isclose(pc2["hit_rate"], 104.0 / 130.0, rel_tol=1e-9), pc2
    cond = pc2["hit_rate_condition"]
    assert "vllm:prefix_cache_queries_total" in cond, cond
    assert "vllm:prompt_tokens_total" in cond, cond
    assert "80.0%*" in live_out, live_out
    assert "vllm:prompt_tokens_total" in live_out, live_out
