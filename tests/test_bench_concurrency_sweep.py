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
    """

    def __init__(self, ids=SERVED, metrics=None, status=None, tokens=8,
                 token_delay=0.002, preemption_step=1.0):
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
