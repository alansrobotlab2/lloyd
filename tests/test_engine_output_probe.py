"""The engine output-integrity probe (#1268): corpus, run record, floor,
comparator and the preemption arm's guard. Every engine is a stub — nothing
here reaches :8096."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path

import pytest

import eval.engine_output_probe as probe

CACHE_INFO = ('vllm:cache_config_info{block_size="3200",cache_dtype="fp8",engine="0",'
              'kv_cache_size_tokens="844969",mamba_cache_mode="align"} 1.0\n')


class FakeEngine:
    """/metrics, /version, /v1/models and /v1/completions, deterministic per
    prompt. `running` and `preemptions` are what the gauges report."""

    base_url = "http://127.0.0.1:8096"

    def __init__(self, *, running: float = 0.0, preemptions: float = 0.0, shape: str = "legacy",
                 cache_info: bool = True, fail: bool = False):
        self.running = running
        self.preemptions = preemptions
        self.shape = shape
        self.cache_info = cache_info
        self.fail = fail
        self.bodies: list[dict] = []
        self._lock = threading.Lock()

    def get_text(self, path: str) -> str:
        if self.fail:
            raise ConnectionError("connection refused")
        if path == "/metrics":
            return ("# HELP x\n"
                    f'vllm:num_requests_running{{engine="0"}} {self.running}\n'
                    'vllm:num_requests_waiting{engine="0"} 0.0\n'
                    f'vllm:num_preemptions_total{{engine="0"}} {self.preemptions}\n'
                    'vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
                    + (CACHE_INFO if self.cache_info else ""))
        if path == "/version":
            return json.dumps({"version": "0.2.1.dev19+gdff1bde84"})
        if path == "/v1/models":
            return json.dumps({"data": [{"id": "Qwen3.8-Flash-Next-nvfp4", "root": "/models/flash"}]})
        raise AssertionError(path)

    def post_json(self, path: str, body: dict) -> dict:
        assert path == "/v1/completions"
        if self.fail:
            raise ConnectionError("connection refused")
        with self._lock:
            self.bodies.append(body)
        seed = int(hashlib.sha256(body["prompt"].encode()).hexdigest()[:6], 16)
        n = min(body["max_tokens"], 6)
        tokens = [f"t{(seed + i) % 97}" for i in range(n)]
        lps = [-((seed + i) % 13) / 10.0 for i in range(n)]
        if self.shape == "legacy":
            lp = {"tokens": tokens, "token_logprobs": lps, "top_logprobs": [], "text_offset": []}
        elif self.shape == "content":
            lp = {"content": [{"token": t, "logprob": v} for t, v in zip(tokens, lps)]}
        else:
            lp = None
        return {"choices": [{"text": "".join(tokens), "finish_reason": "length", "logprobs": lp}],
                "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}}


@pytest.fixture(scope="module")
def corpus():
    return probe.load_corpus()


# --- clause 1: the corpus and the run record --------------------------------

def test_corpus_has_twenty_prompts_with_the_required_shapes(corpus):
    prompts = corpus["prompts"]
    assert len(prompts) >= 20
    by_tag = lambda tag: [p for p in prompts if tag in p["tags"]]  # noqa: E731
    shared = by_tag("shared_prefix")
    assert len(shared) >= 2
    # one long system prompt, byte-identical across the prompts that reuse it
    head = shared[0]["text"].split("<|im_start|>user")[0]
    assert len(head) > 10_000
    assert all(p["text"].startswith(head) for p in shared)
    tools = by_tag("tool_xml")
    assert tools and all("<tools>" in p["text"] and "<tool_call>" in p["text"] for p in tools)
    assert len(by_tag("short")) >= 3
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert len(corpus["sha256"]) == 64


def test_run_sends_temperature_zero_logprobs_one_and_keeps_tokens(corpus):
    eng = FakeEngine()
    rec = probe.run_probe(eng, corpus, label="t")
    assert len(eng.bodies) == len(corpus["prompts"])
    assert all(b["temperature"] == 0 and b["logprobs"] == 1 for b in eng.bodies)
    assert len(rec["prompts"]) == len(corpus["prompts"])
    row = rec["prompts"][0]
    assert row["tokens"] and len(row["tokens"]) == len(row["token_logprobs"])
    assert all(isinstance(v, float) for v in row["token_logprobs"])
    assert rec["corpus"]["sha256"] == corpus["sha256"]
    json.dumps(rec)  # one JSON record


def test_parser_reads_the_legacy_shape_and_falls_back_to_content():
    legacy = FakeEngine(shape="legacy").post_json("/v1/completions", {"prompt": "x", "max_tokens": 4})
    content = FakeEngine(shape="content").post_json("/v1/completions", {"prompt": "x", "max_tokens": 4})
    a, b = probe.parse_completion(legacy), probe.parse_completion(content)
    assert a["tokens"] == b["tokens"] and a["token_logprobs"] == b["token_logprobs"]
    none = FakeEngine(shape="none").post_json("/v1/completions", {"prompt": "x", "max_tokens": 4})
    with pytest.raises(probe.ProbeRefused, match="no readable logprobs"):
        probe.parse_completion(none)


# --- clause 2: the record names its engine; unreachable exits 2 -------------

def test_record_names_the_engine_from_metrics(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "find_engine_venv", lambda port, **_: "/venvs/vllm-flash-next-main")
    rec = probe.run_probe(FakeEngine(), corpus, label="t")
    eng = rec["engine"]
    assert eng["venv"] == "/venvs/vllm-flash-next-main"
    assert eng["vllm_version"] == "0.2.1.dev19+gdff1bde84"
    assert eng["kv_cache_dtype"] == "fp8"
    assert eng["kv_cache_size_tokens"] == 844969
    assert eng["mamba_cache_mode"] == "align"


def test_venv_is_read_off_the_process_serving_the_port(tmp_path):
    proc = tmp_path / "proc"
    for pid, argv in (("11", ["/usr/bin/python", "x.py"]),
                      ("22", ["/v/vllm-djev/bin/python", "-m", "vllm", "--port", "8010"]),
                      ("33", ["/v/vllm-flash-next-main/bin/python", "-m",
                              "vllm.entrypoints.openai.api_server", "--port", "8096", "--host", "h"])):
        (proc / pid).mkdir(parents=True)
        (proc / pid / "cmdline").write_bytes(b"\x00".join(a.encode() for a in argv) + b"\x00")
    assert probe.find_engine_venv(8096, proc_root=proc) == "/v/vllm-flash-next-main"
    assert probe.find_engine_venv(9999, proc_root=proc) is None


def test_unreachable_engine_exits_two_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "runs"
    rc = probe.main(["run", "--out-dir", str(out)], client_factory=lambda url: FakeEngine(fail=True))
    assert rc == 2
    assert "unreachable" in capsys.readouterr().err
    assert not out.exists() or not list(out.iterdir())


def test_run_writes_a_record_that_exits_zero(tmp_path):
    out = tmp_path / "runs"
    rc = probe.main(["run", "--label", "ok", "--out-dir", str(out)], client_factory=lambda url: FakeEngine())
    assert rc == 0
    (path,) = list(out.glob("*_ok.json"))
    assert json.loads(path.read_text())["kind"] == "engine_output_probe"


# --- clause 3: the floor is a file, and without it nothing is decided -------

def _runs(corpus, n=5):
    return [probe.run_probe(FakeEngine(), corpus, label=f"idle-{i}") for i in range(n)]


def test_floor_records_per_prompt_agreement_and_spread(corpus):
    runs = _runs(corpus)
    runs[3]["prompts"][0]["token_logprobs"][1] += 0.02  # one run's jitter
    floor = probe.build_floor(runs)
    assert floor["n_runs"] == 5
    row = floor["prompts"][corpus["prompts"][0]["id"]]
    assert row["pairs"] == 10
    assert row["agreement_min"] == 1.0
    assert row["token_lp_delta_max"] == pytest.approx(0.02)
    assert floor["summary"]["prompts_bitwise_reproducible"] == len(corpus["prompts"]) - 1


def _jittery_runs(corpus, n=5):
    """Prompt 0 is not reproducible idle: its later tokens and every logprob
    move from run to run by a bounded amount, as the long prompts do live."""
    runs = _runs(corpus, n)
    for i, run in enumerate(runs):
        row = run["prompts"][0]
        row["token_logprobs"] = [v - 0.01 * ((i * 7 + j) % 5) for j, v in enumerate(row["token_logprobs"])]
        if i % 2:
            row["tokens"][4] = f"alt{i}"
    return runs


def test_a_prompt_that_jitters_idle_is_decided_on_the_measured_margin(corpus):
    runs = _jittery_runs(corpus)
    floor = probe.build_floor(runs)
    pid = corpus["prompts"][0]["id"]
    assert floor["prompts"][pid]["tier"] == "jitter"
    assert floor["prompts"][corpus["prompts"][1]["id"]]["tier"] == "exact"
    assert floor["jitter_loo_worst_ratio"] is not None and floor["jitter_margin"] >= 1.0
    # a later-token divergence of the kind idle runs produce is not a verdict
    cur = copy.deepcopy(runs[0])
    cur["prompts"][0]["tokens"][4] = "idle-noise"
    assert probe.compare_records(runs[1], cur, floor)["diverged"] == 0
    # a logprob shift past margin x floor on an agreed token is
    cur["prompts"][0]["token_logprobs"][0] -= 2.0
    worst = probe.compare_records(runs[1], cur, floor)["rows"][0]
    assert worst["id"] == pid and worst["exceeds"] and worst["tier"] == "jitter"
    # a token flip on a jitter prompt is NOT decidable (a held-out idle run
    # flipped a first token live); it is reported, not called divergent
    cur = copy.deepcopy(runs[0])
    cur["prompts"][0]["tokens"][0] = "FLIPPED"
    row = next(r for r in probe.compare_records(runs[1], cur, floor)["rows"] if r["id"] == pid)
    assert row["first_divergence"] == 0 and not row["exceeds"]


def test_held_out_idle_runs_never_trip_the_floor(corpus):
    """Leave-one-out over unchanged-engine runs: a floor built without a run
    must not call that run divergent — the false-positive rail."""
    runs = _jittery_runs(corpus, 6)
    for i, held in enumerate(runs):
        rest = runs[:i] + runs[i + 1:]
        floor = probe.build_floor(rest)
        for ref in rest:
            assert probe.compare_records(ref, held, floor)["diverged"] == 0


def test_floor_refuses_runs_from_different_engines(corpus):
    runs = _runs(corpus, 2)
    runs[1]["engine"]["kv_cache_dtype"] = "auto"
    with pytest.raises(probe.ProbeRefused, match="different engines"):
        probe.build_floor(runs)


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


def test_compare_without_a_floor_file_exits_two(corpus, tmp_path, capsys):
    ref, cur = _runs(corpus, 2)
    rc = probe.main(["compare", "--current", str(_write(tmp_path / "c.json", cur)),
                     "--reference", str(_write(tmp_path / "r.json", ref)),
                     "--floor", str(tmp_path / "missing.json")])
    assert rc == 2
    assert "no floor file" in capsys.readouterr().err


def test_the_threshold_is_the_floor_file_not_a_constant(corpus, tmp_path):
    runs = _runs(corpus)
    ref = runs[0]
    cur = copy.deepcopy(runs[1])
    cur["prompts"][2]["token_logprobs"][0] += 0.05
    floor = probe.build_floor(runs)
    rpath, cpath = _write(tmp_path / "r.json", ref), _write(tmp_path / "c.json", cur)
    fpath = _write(tmp_path / "floor.json", floor)
    assert probe.main(["compare", "--current", str(cpath), "--reference", str(rpath),
                       "--floor", str(fpath)]) == 1
    # the same delta, under a floor that measured that much idle spread, passes
    floor["prompts"][ref["prompts"][2]["id"]]["token_lp_delta_max"] = 0.06
    floor["prompts"][ref["prompts"][2]["id"]]["median_lp_delta_max"] = 0.06
    _write(fpath, floor)
    assert probe.main(["compare", "--current", str(cpath), "--reference", str(rpath),
                       "--floor", str(fpath)]) == 0


# --- clause 4: the comparator sees one flipped token and one shifted logprob -

def test_one_flipped_token_is_past_the_floor_with_its_index(corpus):
    runs = _runs(corpus)
    floor = probe.build_floor(runs)
    cur = copy.deepcopy(runs[0])
    target = cur["prompts"][5]
    target["tokens"][3] = "FLIPPED"
    result = probe.compare_records(runs[0], cur, floor)
    assert result["diverged"] == 1
    worst = result["rows"][0]
    assert worst["id"] == target["id"] and worst["exceeds"]
    assert worst["first_divergence"] == 3
    assert worst["agreement"] == pytest.approx(3 / len(target["tokens"]))
    assert all(not r["exceeds"] and r["first_divergence"] is None for r in result["rows"][1:])


def test_one_shifted_logprob_is_past_the_floor_even_when_the_median_holds(corpus):
    runs = _runs(corpus)
    floor = probe.build_floor(runs)
    cur = copy.deepcopy(runs[0])
    target = cur["prompts"][7]
    target["token_logprobs"][-1] -= 0.5  # the minimum moves, the median need not
    result = probe.compare_records(runs[0], cur, floor)
    worst = result["rows"][0]
    assert worst["id"] == target["id"] and worst["exceeds"]
    assert worst["first_divergence"] is None
    assert worst["token_lp_delta"] == pytest.approx(0.5)
    assert any("token logprob delta" in r for r in worst["reasons"])


def test_report_prints_the_worst_prompt_first(corpus):
    runs = _runs(corpus)
    floor = probe.build_floor(runs)
    cur = copy.deepcopy(runs[0])
    cur["prompts"][-1]["tokens"][0] = "X"
    text = probe.render_comparison(probe.compare_records(runs[0], cur, floor))
    rows = [ln for ln in text.splitlines() if ln.startswith(tuple(p["id"] for p in corpus["prompts"]))]
    assert rows[0].startswith(cur["prompts"][-1]["id"]) and "PAST FLOOR" in rows[0]
    assert "1 of" in text


def test_compare_refuses_different_corpora(corpus):
    runs = _runs(corpus, 2)
    floor = probe.build_floor(runs)
    other = copy.deepcopy(runs[1])
    other["corpus"]["sha256"] = "0" * 64
    with pytest.raises(probe.ProbeRefused, match="different corpora"):
        probe.compare_records(runs[0], other, floor)


# --- clause 5: the preemption arm refuses a busy engine and states the result

def test_preempt_refuses_unless_the_engine_is_idle(corpus):
    eng = FakeEngine(running=1.0)
    with pytest.raises(probe.ProbeRefused, match="engine busy"):
        probe.run_preempt(eng, corpus, label="p", sleep=lambda s: None)
    assert eng.bodies == []  # not one load request was sent


def test_preempt_refusal_is_exit_two(corpus, tmp_path):
    rc = probe.main(["preempt", "--out-dir", str(tmp_path)],
                    client_factory=lambda url: FakeEngine(running=2.0))
    assert rc == 2
    assert not list(tmp_path.iterdir())


def test_preempt_not_reached_is_a_stated_result(corpus):
    eng = FakeEngine()
    rec = probe.run_preempt(eng, corpus, label="p", load_requests=2, load_prompt_words=50,
                            load_max_tokens=8, settle_s=0, poll_s=0.01, sleep=lambda s: None)
    assert rec["num_preemptions_before"] == 0.0
    assert rec["num_preemptions_after"] == 0.0
    assert rec["preemptions_reached"] is False
    assert rec["probe"] and len(rec["probe"]["prompts"]) == len(corpus["prompts"])
    loads = [b for b in eng.bodies if b.get("ignore_eos")]
    assert len(loads) == 2 and len({b["prompt"] for b in loads}) == 2


def test_preempt_reached_when_the_counter_moves(corpus):
    eng = FakeEngine()
    real_post = eng.post_json

    def post(path, body):
        if body.get("ignore_eos"):
            eng.preemptions = 3.0  # the engine preempted while the load ran
        return real_post(path, body)

    eng.post_json = post
    rec = probe.run_preempt(eng, corpus, label="p", load_requests=1, load_prompt_words=20,
                            load_max_tokens=4, settle_s=0, poll_s=0.01, sleep=lambda s: None)
    assert rec["preemptions_reached"] is True
    assert (rec["num_preemptions_before"], rec["num_preemptions_after"]) == (0.0, 3.0)


# --- the committed floor ------------------------------------------------------

def test_committed_floor_matches_the_committed_corpus(corpus):
    floor = json.loads(probe.DEFAULT_FLOOR.read_text())
    assert floor["n_runs"] >= 5
    assert floor["corpus_sha256"] == corpus["sha256"]
    assert set(floor["prompts"]) == {p["id"] for p in corpus["prompts"]}
